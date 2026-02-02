mod raw;

pub use crate::raw::{Config, ModernBertModel};
use fusor_core::{Device, FloatDataType, Result, Tensor, VarBuilder};
use tokenizers::Tokenizer;

/// A builder for a [`ModernBert`] model
#[derive(Default)]
pub struct ModernBertBuilder {
    source: ModernBertSource,
    cache: kalosm_common::Cache,
}

pub struct ModernBert {
    embedding_search_prefix: Arc<Option<String>>,
    model: Arc<ModernBertModel>,
    tokenizer: Arc<RwLock<Tokenizer>>,
}

impl ModernBertBuilder {
    /// Set the source of the model
    pub fn with_source(mut self, source: ModernBertSource) -> Self {
        self.source = source;
        self
    }

    /// Build the model
    pub async fn build(self) -> Result<ModernBert, ModernBertLoadingError> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }

    #[cfg(feature = "tokio")]
    /// Set the cache location to use for the model (defaults DATA_DIR/kalosm/cache)
    pub fn with_cache(mut self, cache: kalosm_common::Cache) -> Self {
        self.cache = cache;

        self
    }

    pub async fn build_with_loading_handler(
        self,
        loading_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> Result<ModernBert, ModernBertLoadingError> {
        ModernBert::from_builder(self, loading_handler).await
    }
}

/// An error that can occur when loading a ModernBert model.
#[derive(Debug, thiserror::Error)]
pub enum ModernBertLoadingError {
    /// An error that can occur when trying to load a ModernBert model from huggingface or a local file.
    #[error("Failed to load model from huggingface or local file: {0}")]
    DownloadingError(#[from] CacheError),
    /// An error that can occur when trying to load a ModernBert model.
    #[error("Failed to load model into device: {0}")]
    LoadModel(#[from] fusor_core::Error),
    /// An IO error that can occur when trying to load a bert model.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// An error that can occur when trying to load the bert tokenizer.
    #[error("Failed to load tokenizer: {0}")]
    LoadTokenizer(tokenizers::Error),
    /// An error that can occur when trying to load the bert config.
    #[error("Failed to load config: {0}")]
    LoadConfig(serde_json::Error),
    /// A config was not found
    #[error("Config not found")]
    ConfigNotFound,
}

/// An error that can occur when running a ModernBert model.
#[derive(Debug, thiserror::Error)]
pub enum ModernBertError {
    /// An error that can occur when trying to run a ModernBert model.
    #[error("Failed to run model: {0}")]
    Fusor(#[from] fusor_core::Error),
    /// An error that can occur when tokenizing or detokenizing text.
    #[error("Failed to tokenize: {0}")]
    TokenizerError(tokenizers::Error),
}

impl ModernBert {
    /// Create a new [`ModernBertBuilder`]
    pub fn builder() -> ModernBertBuilder {
        ModernBertBuilder::default()
    }

    /// Create a new default bert model
    pub async fn new() -> Result<Self, ModernBertLoadingError> {
        Self::builder().build().await
    }

    /// Create a new default bert model for search
    pub async fn new_for_search() -> Result<Self, ModernBertLoadingError> {
        Self::builder()
            .with_source(ModernBertSource::new_for_search())
            .build()
            .await
    }

    async fn from_builder(
        builder: ModernBertBuilder,
        mut progress_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> Result<Self, ModernBertLoadingError> {
        let ModernBertBuilder { source, cache } = builder;
        let ModernBertSource {
            config,
            tokenizer,
            model,
            search_embedding_prefix,
        } = source;

        let source = format!("Config ({config})");
        let mut create_progress = ModelLoadingProgress::downloading_progress(source);
        let config = cache
            .get_bytes(&config, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;
        let tokenizer_source = format!("Tokenizer ({tokenizer})");
        let mut create_progress = ModelLoadingProgress::downloading_progress(tokenizer_source);
        let tokenizer_bytes = cache
            .get_bytes(&tokenizer, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;
        let model_source = format!("Model ({model})");
        let mut create_progress = ModelLoadingProgress::downloading_progress(model_source);
        let weights_bytes = cache
            .get_bytes(&model, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        let config: Config =
            serde_json::from_slice(&config).map_err(ModernBertLoadingError::LoadConfig)?;

        let device = Device::new().await?;
        let mut weights = std::io::Cursor::new(&weights_bytes);
        let mut vb = VarBuilder::from_gguf(&mut weights)
            .map_err(|err| ModernBertLoadingError::LoadModel(err.into()))?;

        let model = ModernBertModel::load(&device, &mut vb, &config)?;

        let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(ModernBertLoadingError::LoadTokenizer)?;
        tokenizer.with_padding(None);
        Ok(Self {
            tokenizer: Arc::new(RwLock::new(tokenizer)),
            model: Arc::new(model),
            embedding_search_prefix: Arc::new(search_embedding_prefix),
        })
    }

    pub fn embed(&self, texts: Vec<String>) -> Result<Tensor> {
        let (input_ids, mask) = self.tokenize(&texts)?;
        let hidden = self.encode_tokens(&input_ids, &mask)?;
        let pooled = mean_pool(&hidden, &mask)?;
        normalize(pooled)
    }
}

fn tokenize(&self, texts: &[String]) -> Result<(Tensor, Tensor)> {
    let encodings = self
        .tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let max_len = encodings.iter().map(|e| e.len()).max().unwrap();

    let input_ids: Vec<Vec<u32>> = encodings
        .iter()
        .map(|e| {
            let mut ids = e.get_ids().to_vec();
            ids.resize(max_len, 0);
            ids
        })
        .collect();

    let attention_mask: Vec<Vec<u32>> = encodings
        .iter()
        .map(|e| {
            let mut mask = vec![1; e.len()];
            mask.resize(max_len, 0);
            mask
        })
        .collect();

    let input_ids = Tensor::from_vec(
        input_ids.into_iter().flatten().collect(),
        (encodings.len(), max_len),
        &self.device,
    )?;

    let attention_mask = Tensor::from_vec(
        attention_mask.into_iter().flatten().collect(),
        (encodings.len(), max_len),
        &self.device,
    )?;

    Ok((input_ids, attention_mask))
}

fn encode_tokens(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let output = self.model.forward(input_ids, Some(attention_mask), None)?;

    Ok(output.hidden_state)
}

fn mean_pool(hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let mask = mask.unsqueeze(2)?;
    let masked = hidden * &mask;

    let sum = masked.sum(1)?;
    let count = mask.sum(1)?.clamp_min(1.0)?;

    Ok(sum / count)
}

fn normalize(x: Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum(1)?.sqrt()?.unsqueeze(1)?;
    Ok(x / norm)
}
