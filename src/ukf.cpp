#include <iostream>
#include "ukf.h"

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, first measurement will be used to initialize state
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initalise previous timestamp
  previous_timestamp_ = 0.0;

  // initialize size of states and spreading parameter
  n_x_ = 5;
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);
  x_.fill(0.0);
  x_aug_ = VectorXd(n_aug_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_.fill(0.0);
  P_aug_ = MatrixXd(n_aug_, n_aug_);
  P_aug_.fill(0.0);

  // initial predictied sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);
  Xsig_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_.fill(0.0);

  // Set weights
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for(int i=1; i < 2*n_aug_+1; i++)
  {
    weights_(i) = 0.5/(lambda_ + n_aug_);
  }

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Laser covariance
  R_laser_ = MatrixXd(2,2);
  R_laser_ << std_laspx_*std_laspx_, 0.0,
              0, std_laspy_*std_laspy_;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.35;

  // Radar covariance
  R_radar_ = MatrixXd(3,3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0, std_radrd_*std_radrd_;

  // initialize NIS variables
  NIS_radar_ = 0.0;
  NIS_laser_ = 0.0;

}

UKF::~UKF() {}

/**
 * InitializeState intializes the UKF state by the first measurement
 * @param meas_package The latest measurement data of either radar or laser
 */
void UKF::InitializeState(MeasurementPackage meas_package) {

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    double rho = meas_package.raw_measurements_[0];
    double phi = meas_package.raw_measurements_[1];
    double ro_dot = meas_package.raw_measurements_[2];

    double p_x = rho * cos(phi);
    double p_y = rho * sin(phi);

    x_(0) = p_x;
    x_(1) = p_y;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    double p_x = meas_package.raw_measurements_[0];
    double p_y = meas_package.raw_measurements_[1];

    x_(0) = p_x;
    x_(1) = p_y;
  }

  P_ <<   1, 0,  0,  0,  0,
          0,  1, 0,  0,  0,
          0,  0,  1, 0,  0,
          0,  0,  0,  1, 0,
          0,  0,  0,  0,  1;
}
/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if(is_initialized_ == false){
    InitializeState(meas_package);
    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = meas_package.timestamp_;

  GenerateSigmaPoints();
  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

/**
 * GenerateSigmaPoints gerenates sigma points to be used in the Prdiction
 * step
 */
void UKF::GenerateSigmaPoints() {

  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  P_aug_.topLeftCorner(5,5) = P_;
  P_aug_(5,5) = pow(std_a_,2);
  P_aug_(6,6) = pow(std_yawdd_,2);
  MatrixXd A = P_aug_.llt().matrixL();

  // Generate sigma points
  Xsig_.col(0) = x_aug_;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_.col(i+1) = x_aug_ + sqrt(lambda_+n_aug_) * A.col(i);
    Xsig_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * A.col(i);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  // Predict for sigma points
  for (int i = 0; i < 2*n_aug_+1; i++)
  {
    double p_x = Xsig_(0,i);
    double p_y = Xsig_(1,i);
    double v = Xsig_(2,i);
    double yaw = Xsig_(3,i);
    double yawd = Xsig_(4,i);
    double nu_a = Xsig_(5,i);
    double nu_yawdd = Xsig_(6,i);

    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  // update mean
  x_.fill(0.0);
  for(int i=0; i < 2*n_aug_ + 1; i++)
  {
    x_ = x_ + weights_(i)*Xsig_pred_.col(i);
  }
  // update covaraince
  P_.fill(0.0);
  for(int i=0; i<2*n_aug_ +1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i)*x_diff*x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  MatrixXd Zsig_ = MatrixXd(2, 2*n_aug_ + 1);
  // Transform sigma points to measurement space
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    Zsig_.col(i) << p_x,
                    p_y;
  }

  // calculate mean
  VectorXd z_pred = VectorXd(2);
  z_pred.fill(0.0);
  for(int i =0; i< 2*n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i)*Zsig_.col(i);
  }

  // calculate covariance and cross-correlation
  MatrixXd S = MatrixXd(2,2);
  S.fill(0.0);
  MatrixXd T = MatrixXd(n_x_, 2);
  T.fill(0.0);

  for(int i=0; i < 2*n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig_.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
    while(z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    S = S + weights_(i)*z_diff*z_diff.transpose();
    T = T + weights_(i)*x_diff*z_diff.transpose();
  }
  S = S + R_laser_;

  // calculate Kalman gain and update state
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  MatrixXd K = T*S.inverse();
  x_ = x_ + K * y;
  P_ = P_ - K*S*K.transpose();

  // calculate NIS
  NIS_laser_ = y.transpose() * S.inverse() * y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  MatrixXd Zsig_ = MatrixXd(3, 2*n_aug_ + 1);
  // Transform sigma points to measurement space
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double yawd = Xsig_pred_(4,i);

    if(fabs(p_x) <= 0.001){
          p_x = 0.001;
    }
    if(fabs(p_y) <= 0.001){
          p_y = 0.001;
    }

    double rho = sqrt(pow(p_x, 2) + pow(p_y, 2));
    double phi =atan2(p_y,p_x);
    double rho_dot = (p_x*cos(yaw)*v+p_y*sin(yaw)*v) / rho;

    Zsig_.col(i) << rho,
                    phi,
                    rho_dot;
  }

  // calculate mean
  VectorXd z_pred = VectorXd(3);
  z_pred.fill(0.0);
  for(int i =0; i< 2*n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i)*Zsig_.col(i);
  }

  // calculate covariance and cross-correlation
  MatrixXd S = MatrixXd(3,3);
  S.fill(0.0);
  MatrixXd T = MatrixXd(n_x_, 3);
  T.fill(0.0);

  for(int i=0; i < 2*n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig_.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    while(z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
    while(z_diff(1) <-M_PI) z_diff(1) += 2.*M_PI;

    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    S = S + weights_(i)*z_diff*z_diff.transpose();
    T = T + weights_(i)*x_diff*z_diff.transpose();

  }
  S = S + R_radar_;

  // calculate Kalman gain and update state
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  MatrixXd K = T*S.inverse();
  // angle normalization
  while (y(1)> M_PI) y(1)-=2.*M_PI;
  while (y(1)<-M_PI) y(1)+=2.*M_PI;
  x_ = x_ + K * y;
  P_ = P_ - K*S*K.transpose();

  // calculate NIS
  NIS_radar_ = y.transpose() * S.inverse() * y;
}
