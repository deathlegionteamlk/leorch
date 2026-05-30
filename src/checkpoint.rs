use crate::tensor::{Tensor, TensorDtype};
use crate::error::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Checkpoint {
    model_state: HashMap<String, Tensor>,
    optimizer_state: HashMap<String, Vec<Tensor>>,
    epoch: usize,
    global_step: usize,
    best_metric: Option<TensorDtype>,
    extra_data: HashMap<String, String>,
}

impl Checkpoint {
    pub fn new(
        model_state: HashMap<String, Tensor>,
        optimizer_state: HashMap<String, Vec<Tensor>>,
        epoch: usize,
        global_step: usize,
    ) -> Self {
        Self {
            model_state,
            optimizer_state,
            epoch,
            global_step,
            best_metric: None,
            extra_data: HashMap::new(),
        }
    }
    
    pub fn with_best_metric(mut self, metric: TensorDtype) -> Self {
        self.best_metric = Some(metric);
        self
    }
    
    pub fn with_extra_data(mut self, key: &str, value: &str) -> Self {
        self.extra_data.insert(key.to_string(), value.to_string());
        self
    }
    
    pub fn epoch(&self) -> usize {
        self.epoch
    }
    
    pub fn global_step(&self) -> usize {
        self.global_step
    }
    
    pub fn model_state(&self) -> &HashMap<String, Tensor> {
        &self.model_state
    }
    
    pub fn optimizer_state(&self) -> &HashMap<String, Vec<Tensor>> {
        &self.optimizer_state
    }
    
    pub fn save(&self, path: &str) -> Result<()> {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let serialized = format!(
            "epoch:{}\nglobal_step:{}\nbest_metric:{:?}\n",
            self.epoch, self.global_step, self.best_metric
        );
        
        fs::write(path, serialized)?;
        
        Ok(())
    }
    
    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        
        let mut epoch = 0;
        let mut global_step = 0;
        let mut best_metric = None;
        
        for line in content.lines() {
            if line.starts_with("epoch:") {
                epoch = line[6..].parse().unwrap_or(0);
            } else if line.starts_with("global_step:") {
                global_step = line[12..].parse().unwrap_or(0);
            }
        }
        
        Ok(Self {
            model_state: HashMap::new(),
            optimizer_state: HashMap::new(),
            epoch,
            global_step,
            best_metric,
            extra_data: HashMap::new(),
        })
    }
}

pub struct CheckpointManager {
    checkpoint_dir: String,
    max_checkpoints: usize,
    keep_best: bool,
    checkpoints: Vec<String>,
}

impl CheckpointManager {
    pub fn new(checkpoint_dir: &str, max_checkpoints: usize) -> Self {
        Self::with_options(checkpoint_dir, max_checkpoints, true)
    }
    
    pub fn with_options(
        checkpoint_dir: &str,
        max_checkpoints: usize,
        keep_best: bool,
    ) -> Self {
        fs::create_dir_all(checkpoint_dir).ok();
        
        Self {
            checkpoint_dir: checkpoint_dir.to_string(),
            max_checkpoints,
            keep_best,
            checkpoints: vec![],
        }
    }
    
    pub fn save(&mut self, checkpoint: &Checkpoint, is_best: bool) -> Result<String> {
        let filename = format!("checkpoint_epoch_{}.ckpt", checkpoint.epoch());
        let path = Path::new(&self.checkpoint_dir).join(&filename);
        
        checkpoint.save(path.to_str().unwrap())?;
        
        self.checkpoints.push(filename.clone());
        
        if self.checkpoints.len() > self.max_checkpoints {
            let to_remove = self.checkpoints.remove(0);
            if !self.keep_best || !to_remove.contains("best") {
                let old_path = Path::new(&self.checkpoint_dir).join(&to_remove);
                fs::remove_file(old_path).ok();
            }
        }
        
        if is_best {
            let best_path = Path::new(&self.checkpoint_dir).join("best_checkpoint.ckpt");
            checkpoint.save(best_path.to_str().unwrap())?;
        }
        
        Ok(path.to_string_lossy().to_string())
    }
    
    pub fn load_latest(&self) -> Option<Checkpoint> {
        self.checkpoints.last()
            .and_then(|f| {
                let path = Path::new(&self.checkpoint_dir).join(f);
                Checkpoint::load(path.to_str().unwrap()).ok()
            })
    }
    
    pub fn load_best(&self) -> Option<Checkpoint> {
        let path = Path::new(&self.checkpoint_dir).join("best_checkpoint.ckpt");
        Checkpoint::load(path.to_str().unwrap()).ok()
    }
    
    pub fn load_by_epoch(&self, epoch: usize) -> Option<Checkpoint> {
        let filename = format!("checkpoint_epoch_{}.ckpt", epoch);
        let path = Path::new(&self.checkpoint_dir).join(filename);
        Checkpoint::load(path.to_str().unwrap()).ok()
    }
    
    pub fn list_checkpoints(&self) -> Vec<String> {
        self.checkpoints.clone()
    }
    
    pub fn cleanup(&self) -> Result<()> {
        for filename in &self.checkpoints {
            let path = Path::new(&self.checkpoint_dir).join(filename);
            fs::remove_file(path).ok();
        }
        Ok(())
    }
}

pub fn save_model(model: &dyn crate::nn::Module, path: &str) -> Result<()> {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    fs::write(path, "model_state")?;
    
    Ok(())
}

pub fn load_model(path: &str) -> Result<HashMap<String, Tensor>> {
    let _content = fs::read_to_string(path)?;
    
    Ok(HashMap::new())
}

pub struct EarlyStopping {
    patience: usize,
    min_delta: TensorDtype,
    counter: usize,
    best_score: Option<TensorDtype>,
    should_stop: bool,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: TensorDtype) -> Self {
        Self {
            patience,
            min_delta,
            counter: 0,
            best_score: None,
            should_stop: false,
        }
    }
    
    pub fn step(&mut self, score: TensorDtype) {
        if let Some(best) = self.best_score {
            if score > best + self.min_delta {
                self.best_score = Some(score);
                self.counter = 0;
            } else {
                self.counter += 1;
                if self.counter >= self.patience {
                    self.should_stop = true;
                }
            }
        } else {
            self.best_score = Some(score);
        }
    }
    
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }
    
    pub fn reset(&mut self) {
        self.counter = 0;
        self.best_score = None;
        self.should_stop = false;
    }
}