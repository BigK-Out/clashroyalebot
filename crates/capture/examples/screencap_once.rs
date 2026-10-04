fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let adb = format!("{}/Android/Sdk/platform-tools/adb", std::env::var("HOME")?);
    let f = capture::screencap(&adb, &a[1], 576)?;
    image::RgbImage::from_raw(f.width, f.height, f.rgb).unwrap().save(&a[2])?;
    println!("{}x{}", f.width, f.height);
    Ok(())
}
