// src/main.rs
use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;

use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::InMemDataset;
use burn::module::Module;
use burn::nn;
use burn::optim::{AdamConfig, Optimizer};

use burn::tensor::{Int, Tensor};

use flate2::read::GzDecoder;
use image::{GenericImageView, Pixel};
use indicatif::ProgressBar;
use reqwest::blocking::get;

/// Backend alias (Autodiff<NdArray>)
type B = Autodiff<NdArray<f32>>;

/// Paths & MNIST sources
const MNIST_BASE: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const TRAIN_IMAGES: &str = "train-images-idx3-ubyte.gz";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte.gz";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte.gz";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte.gz";

/// --- Utility: download file if not exists
fn download_if_needed(fname: &str) -> std::io::Result<PathBuf> {
    let dir = PathBuf::from("training_data");
    fs::create_dir_all(&dir)?;
    let mut path = dir.clone();
    path.push(fname);

    if path.exists() {
        println!("File exists: {:?}", path);
        return Ok(path);
    }

    let url = format!("{MNIST_BASE}{fname}");
    println!("Downloading {url} ...");
    let mut resp =
        get(&url).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let mut out = File::create(&path)?;
    resp.copy_to(&mut out)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    println!("Saved to {:?}", path);
    Ok(path)
}

/// parse IDX ubyte for images -> Vec<Vec<u8>> with each image flattened
fn parse_idx_images(path_gz: &PathBuf) -> std::io::Result<Vec<Vec<u8>>> {
    let f = File::open(path_gz)?;
    let mut gz = GzDecoder::new(f);
    let mut buf = vec![];
    gz.read_to_end(&mut buf)?;
    // idx header: magic (4 bytes), n_items (4), rows (4), cols (4)
    if buf.len() < 16 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "bad idx images",
        ));
    }
    let n = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let expected = 16 + n * rows * cols;
    if buf.len() < expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "idx image truncated",
        ));
    }
    let mut res = Vec::with_capacity(n);
    let mut offset = 16;
    for _ in 0..n {
        let slice = &buf[offset..offset + rows * cols];
        res.push(slice.to_vec());
        offset += rows * cols;
    }
    Ok(res)
}

/// parse IDX labels -> Vec<u8>
fn parse_idx_labels(path_gz: &PathBuf) -> std::io::Result<Vec<u8>> {
    let f = File::open(path_gz)?;
    let mut gz = GzDecoder::new(f);
    let mut buf = vec![];
    gz.read_to_end(&mut buf)?;
    if buf.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "bad idx labels",
        ));
    }
    let n = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    if buf.len() < 8 + n {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "idx labels truncated",
        ));
    }
    Ok(buf[8..8 + n].to_vec())
}

/// Convert image bytes (0..255) -> float32 normalized 0.0..1.0
fn images_to_tensors(
    images: Vec<Vec<u8>>,
    device: &<B as burn::tensor::backend::Backend>::Device,
) -> Vec<Tensor<B, 3>> {
    // Each image -> Tensor shape [1, H, W] (channel-first)
    let mut out = Vec::with_capacity(images.len());
    for img in images {
        let floats: Vec<f32> = img.into_iter().map(|b| (b as f32) / 255.0).collect();
        // Create Tensor<B, 1> then reshape to [1, H, W]
        let t: Tensor<B, 1> = Tensor::from_floats(floats.as_slice(), device);
        // assume 28x28
        let t = t.reshape([1, 28, 28]);
        out.push(t);
    }
    out
}

/// Build InMemDataset of tuples (image_tensor, label_int)
fn build_inmem_dataset(
    images: Vec<Tensor<B, 3>>,
    labels: Vec<u8>,
) -> InMemDataset<(Tensor<B, 4>, Tensor<B, 1, Int>)> {
    // We will store flattened tensors and create mapper that expands to batch shape
    // But InMemDataset in Burn can store arbitrary items — we'll store (image, label)
    let mut items: Vec<(Tensor<B, 4>, Tensor<B, 1, Int>)> = Vec::with_capacity(images.len());
    for (img, &lbl) in images.into_iter().zip(labels.iter()) {
        // reshape to [1, C=1, H, W]
        let img4 = img.reshape([1, 1, 28, 28]);
        // label to Tensor<B,1,Int>
        let lbl_t: Tensor<B, 1, Int> = Tensor::from_ints(
            [lbl as i64],
            &<B as burn::tensor::backend::Backend>::Device::default(),
        );
        items.push((img4, lbl_t));
    }
    InMemDataset::new(items)
}

/// CNN model: simple conv -> relu -> pool -> flatten -> linear
#[derive(Module, Debug)]
pub struct CnnModel<B: burn::tensor::backend::Backend> {
    conv1: nn::conv::Conv2d<B>,
    relu: nn::activation::Relu,
    pool: nn::pool::MaxPool2d,
    fc: nn::Linear<B>,
}

impl<B: burn::tensor::backend::Backend> CnnModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = nn::conv::Conv2dConfig::new([1, 8], [3, 3])
            .with_padding(nn::PaddingConfig2d::Same)
            .init(device);
        let pool = nn::pool::MaxPool2dConfig::new([2, 2]).init();
        let fc = nn::LinearConfig::new(8 * 14 * 14, 10).init(device);

        Self {
            conv1,
            relu: nn::activation::Relu::new(),
            pool,
            fc,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);
        // Flatten: reshape from [batch, channels, h, w] to [batch, channels*h*w]
        let x = x.flatten(1, 3);
        self.fc.forward(x)
    }
}

fn infer_image(
    model: &CnnModel<B>,
    image_path: &str,
    device: &<B as burn::tensor::backend::Backend>::Device,
) {
    println!("Loading {}...", image_path);
    let img = match image::open(image_path) {
        Ok(i) => i,
        Err(e) => {
            println!("Failed to load image {}: {}", image_path, e);
            return;
        }
    };

    // Resize to 28x28 and grayscale
    let img = img.resize_exact(28, 28, image::imageops::FilterType::Nearest);
    let img = img.grayscale();

    let mut pixels = Vec::new();
    for y in 0..28 {
        for x in 0..28 {
            let p = img.get_pixel(x, y);
            // Normalize 0..1
            let val = p.channels()[0] as f32 / 255.0;
            pixels.push(val);
        }
    }

    // Create tensor [1, 1, 28, 28]
    let input_tensor =
        Tensor::<B, 1>::from_floats(pixels.as_slice(), device).reshape([1, 1, 28, 28]);

    let output = model.forward(input_tensor);
    let predicted = output.argmax(1).into_scalar() as i32;

    println!("Predicted digit for {}: {}", image_path, predicted);
}

fn main() -> anyhow::Result<()> {
    println!("MNIST train (Burn 0.19) - start");

    // device
    let device = <B as burn::tensor::backend::Backend>::Device::default();

    // download files (cache dir)
    let train_images_path = download_if_needed(TRAIN_IMAGES)?;
    let train_labels_path = download_if_needed(TRAIN_LABELS)?;
    let test_images_path = download_if_needed(TEST_IMAGES)?;
    let test_labels_path = download_if_needed(TEST_LABELS)?;

    // parse
    println!("Parsing training images...");
    let train_images_raw = parse_idx_images(&train_images_path)?;
    println!("Parsing training labels...");
    let train_labels_raw = parse_idx_labels(&train_labels_path)?;

    println!("Parsing test images...");
    let test_images_raw = parse_idx_images(&test_images_path)?;
    println!("Parsing test labels...");
    let test_labels_raw = parse_idx_labels(&test_labels_path)?;

    // convert to tensors
    println!("Converting to tensors...");
    let train_imgs_t = images_to_tensors(train_images_raw, &device);
    let test_imgs_t = images_to_tensors(test_images_raw, &device);

    // build in-mem datasets
    println!("Building in-memory datasets...");
    let train_imgs_t: Vec<_> = train_imgs_t.into_iter().take(1000).collect();
    let train_labels_raw: Vec<_> = train_labels_raw.into_iter().take(1000).collect();
    let train_dataset = build_inmem_dataset(train_imgs_t, train_labels_raw);
    let test_dataset = build_inmem_dataset(test_imgs_t, test_labels_raw);

    // create DataLoaders (batching)
    let batch_size = 64usize;

    // Create a simple batcher
    #[derive(Clone)]
    struct TupleBatcher<B: burn::tensor::backend::Backend> {
        _phantom: std::marker::PhantomData<B>,
    }

    impl<B: burn::tensor::backend::Backend>
        Batcher<B, (Tensor<B, 4>, Tensor<B, 1, Int>), (Tensor<B, 4>, Tensor<B, 1, Int>)>
        for TupleBatcher<B>
    {
        fn batch(
            &self,
            items: Vec<(Tensor<B, 4>, Tensor<B, 1, Int>)>,
            _device: &B::Device,
        ) -> (Tensor<B, 4>, Tensor<B, 1, Int>) {
            let images: Vec<Tensor<B, 4>> = items.iter().map(|(img, _)| img.clone()).collect();
            let labels: Vec<Tensor<B, 1, Int>> = items.iter().map(|(_, lbl)| lbl.clone()).collect();

            let images_batch = Tensor::cat(images, 0);
            let labels_batch = Tensor::cat(labels, 0);

            (images_batch, labels_batch)
        }
    }

    let batcher = TupleBatcher::<B> {
        _phantom: std::marker::PhantomData,
    };

    let train_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(0)
        .build(train_dataset);

    let test_loader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .num_workers(0)
        .build(test_dataset);

    println!("Constructing model...");
    let mut model = CnnModel::new(&device);

    // Optimizer
    let config_optimizer = AdamConfig::new();
    let mut optimizer = config_optimizer.init();

    // training loop
    let epochs = 1usize;
    let lr = 1e-3;

    for epoch in 1..=epochs {
        println!("Epoch {}/{}", epoch, epochs);
        let pb = ProgressBar::new(60000 / batch_size as u64); // MNIST train set size

        for batch in train_loader.iter() {
            let (images_batch, labels_batch) = batch;

            // Forward pass
            let logits = model.forward(images_batch.clone());

            // Loss computation using cross entropy
            let labels_squeezed = labels_batch.clone().squeeze::<1>();
            let loss_config = nn::loss::CrossEntropyLossConfig::new();
            let loss_fn = loss_config.init(&device);
            let loss = loss_fn.forward(logits.clone(), labels_squeezed.clone());

            // Backward pass
            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            pb.inc(1);
        }
        pb.finish_with_message("done");

        // Validation pass
        println!("Running validation...");
        let mut correct = 0;
        let mut total = 0;

        for batch in test_loader.iter() {
            let (images_batch, labels_batch) = batch;
            let logits = model.forward(images_batch);
            let predictions = logits.argmax(1).squeeze::<1>();
            let targets = labels_batch.clone().squeeze::<1>();

            let equal = predictions.equal(targets);
            // Count correct (simplified, might need casting)
            let equal_int = equal.int();
            let sum_correct = equal_int.sum().into_scalar();

            correct += sum_correct as usize;
            total += labels_batch.dims()[0];
        }

        println!(
            "Epoch {} completed. Accuracy: {:.2}%",
            epoch,
            100.0 * correct as f64 / total as f64
        );
    }

    println!("Training finished.");

    // Inference
    infer_image(&model, "digit.png", &device);

    Ok(())
}
