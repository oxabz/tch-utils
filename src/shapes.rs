/*!
This module contains functions to generate shapes.
 */

#[cfg(feature = "rayon")]
use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

/**
Generate an ellipse distance field.

# Arguments
- width: i64 - The width of the image
- height: i64 - The height of the image
- center: (f64, f64) - The center of the ellipse
- radii: (f64, f64) - The radii of the ellipse
- angle: f64 - The angle of the ellipse

# Returns
Tensor - The ellipse distance field [1, H, W] tensor with values in [0, 1]
 */
fn ellipse_distance_field(
    width: usize,
    height: usize,
    center: (f64, f64),
    radii: (f64, f64),
    angle: f64,
    device: Device,
) -> Tensor {
    // Create a tensor with the position of each pixel
    let pos = crate::tensor_init::position_tensor_2d((width, height), 1, (Kind::Float, device));
    // Scale the position tensor to the size of the tensor
    let pos = crate::ops_2d::scale_2d(&pos, &[(width / 2) as f64, (height / 2) as f64]);
    // Translate the position tensor to the center of the ellipse
    let pos = crate::ops_2d::translate_2d(&pos, &[-center.1, -center.0]);
    // Rotate the position tensor to the angle of the ellipse
    let pos = crate::ops_2d::rotate_2d(&pos, angle);
    // Scale the position tensor to the size of the ellipse
    let pos = crate::ops_2d::scale_2d(&pos, &[1.0 / radii.1, 1.0 / radii.0]);

    (crate::ops_2d::norm_2d(&pos) - 1.0).view([1, height as i64, width as i64])
}

/**
Generate an ellipse.

# Arguments
- width: usize - The width of the image
- height: usize - The height of the image
- center: (f64, f64) - The center of the ellipse
- radii: (f64, f64) - The radii of the ellipse
- angle: f64 - The angle of the ellipse

# Returns
Tensor - The ellipse [1, H, W] tensor with value resulting from casting a boolean to the given kind
 */
pub fn ellipse(
    width: usize,
    height: usize,
    center: (f64, f64),
    radii: (f64, f64),
    angle: f64,
    options: (Kind, Device),
) -> Tensor {
    let (kind, device) = options;
    let df = ellipse_distance_field(width, height, center, radii, angle, device);
    (df.lt(0.0)).to_kind(kind)
}

/**
Generate a circle. (wrapper around ellipse)

# Arguments
- width: usize - The width of the image
- height: usize - The height of the image
- center: (f64, f64) - The center of the circle
- radius: f64 - The radius of the circle

# Returns
Tensor - The circle [1, H, W] tensor with value resulting from casting a boolean to the given kind
 */
pub fn circle(
    width: usize,
    height: usize,
    center: (f64, f64),
    radius: f64,
    options: (Kind, Device),
) -> Tensor {
    ellipse(width, height, center, (radius, radius), 0.0, options)
}

fn use_segment(seg: Segment, y: f64) -> bool {
    let ((_, y1), (_, y2), _, _) = seg;
    y1 <= y && y2 > y || y1 > y && y2 <= y
}

type Segment = ((f64, f64), (f64, f64), f64, f64);

/**
Generate a mask from a polygon using the scaning method

# Arguments
width - The width of the image
height - The height of the image
polygon - The polygon to generate the mask from coordinates in the form of [(x1, y1), (x2, y2), ...]
        where [0, 0] is the center of the image and the coordinates are in pixels
options - The kind and device to cast the mask to

> Warning: This function can cause a deadlock when called in a rayon thread that uses Mutex. (if the rayon feature is enabled)

 */
pub fn polygon(
    width: usize,
    height: usize,
    polygon: &Vec<(f64, f64)>,
    options: (Kind, Device),
) -> Tensor {
    // Building the segments and storing the parametric representation of each segment
    let mut segments = vec![];
    for i in 0..polygon.len() {
        let p1 = polygon[i];
        let p2 = polygon[(i + 1) % polygon.len()];
        let a = (p1.1 - p2.1) / (p1.0 - p2.0 + 1e-6); // Adding a small value to avoid division by 0 when the segment is vertical
        let b = p1.1 - a * p1.0;
        segments.push((p1, p2, a, b));
    }
    let segments = segments;

    #[cfg(feature = "rayon")]
    let iter = (0..height).into_par_iter();
    #[cfg(not(feature = "rayon"))]
    let iter = (0..height).into_iter();

    let mask = iter
        .map(|y| {
            let y = y as f64 - height as f64 / 2.0;
            let mut intersections: Vec<_> = segments
                .iter()
                .filter(|segment| use_segment(**segment, y))
                .map(|segment| {
                    let ((x1, _), (x2, _), a, b) = segment;
                    let r = if a.abs() < 1e-6 {
                        (*x1 + *x2) / 2.0
                    } else {
                        (y as f64 - b) / a
                    };
                    ((r + width as f64 / 2.0).round() as usize).clamp(0, width -1)
                })
                .collect();
            intersections.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut column = vec![0u8; width];
            for i in (0..intersections.len()).step_by(2) {
                let x1 = intersections[i];
                let x2 = intersections[i + 1];
                for x in x1..x2 {
                    column[x] = 1u8;
                }
            }
            column
        })
        .flatten()
        .collect::<Vec<u8>>();
    let mask = Tensor::of_slice(&mask).view([1, height as i64, width as i64]);
    mask.to_kind(options.0).to_device(options.1)
}


/**
Generate a mask of the convex hull of a set of points

# Arguments
width - The width of the image
height - The height of the image
points - The points to generate the mask from coordinates in the form of [(x1, y1), (x2, y2), ...]
         where [0, 0] is the center of the image and the coordinates are in pixels
options - The kind and device to cast the mask to

> Warning: This function can cause a deadlock when called in a rayon thread that uses Mutex. (if the rayon feature is enabled)
 */
pub fn convex_hull(
    width: usize,
    height: usize,
    points: &Vec<(f64, f64)>,
    options: (Kind, Device),
) -> Tensor {
    let convex_hull = crate::utils::graham_scan(points);
    polygon(width, height, &convex_hull, options)
}

#[cfg(test)]
mod test {
    use super::*;
    use tch::{Device, Kind};
    use crate::utils::assert_tensor_asset;

    #[test]
    fn test_ellipse() {
        let centered_circle = ellipse(
            100,
            100,
            (0.0, 0.0),
            (25.0, 25.0),
            0.0,
            (Kind::Float, Device::Cpu),
        );
        let offset_circle = ellipse(
            100,
            100,
            (20.0, 25.0),
            (20.0, 20.0),
            0.0,
            (Kind::Float, Device::Cpu),
        );
        let ellipse_ = ellipse(
            100,
            100,
            (0.0, 0.0),
            (25.0, 10.0),
            0.0,
            (Kind::Float, Device::Cpu),
        );
        let rotated_ellipse = ellipse(
            100,
            100,
            (0.0, 0.0),
            (25.0, 10.0),
            45.0f64.to_radians(),
            (Kind::Float, Device::Cpu),
        );
        let centered_rotated_ellipse = ellipse(
            100,
            100,
            (-20.0, 0.0),
            (25.0, 10.0),
            45.0f64.to_radians(),
            (Kind::Float, Device::Cpu),
        );

        tch::vision::image::save(
            &(&centered_circle * 255),
            "test-results/centered_circle.png",
        )
        .unwrap();
        tch::vision::image::save(&(&offset_circle * 255), "test-results/offset_circle.png")
            .unwrap();
        tch::vision::image::save(&(&ellipse_ * 255), "test-results/ellipse.png").unwrap();
        tch::vision::image::save(
            &(&rotated_ellipse * 255),
            "test-results/rotated_ellipse.png",
        )
        .unwrap();
        tch::vision::image::save(
            &(&centered_rotated_ellipse * 255),
            "test-results/centered_rotated_ellipse.png",
        )
        .unwrap();

        assert_tensor_asset(&centered_circle, "test-assets/shapes/centered_circle.pt");
        assert_tensor_asset(&offset_circle, "test-assets/shapes/offset_circle.pt");
        assert_tensor_asset(&ellipse_, "test-assets/shapes/ellipse.pt");
        assert_tensor_asset(&rotated_ellipse, "test-assets/shapes/rotated_ellipse.pt");
        assert_tensor_asset(
            &centered_rotated_ellipse,
            "test-assets/shapes/centered_rotated_ellipse.pt",
        );
    }

    #[test]
    fn test_polygon() {
        let triangle = polygon(
            100,
            100,
            &vec![(0.0, 0.0), (0.0, 50.0), (50.0, 0.0)],
            (Kind::Float, Device::Cpu),
        );
        let square = polygon(
            100,
            100,
            &vec![(0.0, 0.0), (0.0, 50.0), (50.0, 50.0), (50.0, 0.0)],
            (Kind::Float, Device::Cpu),
        );
        let pentagon = polygon(
            100,
            100,
            &vec![
                (0.0, 0.0),
                (0.0, 50.0),
                (25.0, 50.0),
                (50.0, 25.0),
                (50.0, 0.0),
            ],
            (Kind::Float, Device::Cpu),
        );

        tch::vision::image::save(&(&triangle * 255), "test-results/triangle.png").unwrap();
        tch::vision::image::save(&(&square * 255), "test-results/square.png").unwrap();
        tch::vision::image::save(&(&pentagon * 255), "test-results/pentagon.png").unwrap();
        assert_tensor_asset(&triangle, "test-assets/shapes/triangle.pt");
        assert_tensor_asset(&square, "test-assets/shapes/square.pt");
        assert_tensor_asset(&pentagon, "test-assets/shapes/pentagon.pt");
    }

    #[test]
    fn test_convex_hull() {
        let square = vec![(0.0, 0.0), (0.0, 50.0), (50.0, 50.0), (50.0, 0.0)];
        let hourglass = vec![
            (-25.0, -25.0),
            (-25.0, 25.0),
            (0.0, 0.0),
            (25.0, 25.0),
            (25.0, -25.0),
        ];

        let square_hull = convex_hull(100, 100, &square, (Kind::Float, Device::Cpu));
        let hourglass_hull = convex_hull(100, 100, &hourglass, (Kind::Float, Device::Cpu));

        tch::vision::image::save(&(&square_hull * 255), "test-results/square_hull.png").unwrap();
        tch::vision::image::save(&(&hourglass_hull * 255), "test-results/hourglass_hull.png")
            .unwrap();

        assert_tensor_asset(&square_hull, "test-assets/shapes/square_hull.pt");
        assert_tensor_asset(&hourglass_hull, "test-assets/shapes/hourglass_hull.pt");
    }
}
