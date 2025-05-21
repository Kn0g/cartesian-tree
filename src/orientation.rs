use nalgebra::UnitQuaternion;

pub trait IntoOrientation {
    fn into_orientation(self) -> UnitQuaternion<f64>;
}

impl IntoOrientation for UnitQuaternion<f64> {
    fn into_orientation(self) -> UnitQuaternion<f64> {
        self
    }
}

impl IntoOrientation for (f64, f64, f64) {
    fn into_orientation(self) -> UnitQuaternion<f64> {
        UnitQuaternion::from_euler_angles(self.0, self.1, self.2)
    }
}
