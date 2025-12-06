use image::GenericImageView;
use image::Pixel;

fn main() {
    let img = image::open("digit.png").expect("Failed to open image");
    let img = img.grayscale();
    let (w, h) = img.dimensions();
    let mut sum = 0.0;
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            sum += p.channels()[0] as f32;
        }
    }
    let avg = sum / (w * h) as f32;
    println!("Average pixel value: {}", avg);
    if avg > 128.0 {
        println!("Image seems to be bright (likely black on white). Needs inversion for MNIST.");
    } else {
        println!("Image seems to be dark (likely white on black). Matches MNIST.");
    }
}
