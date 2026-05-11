The dataset has the following file structure:
- patrol_ship_ood/processed/test/*.tab
- patrol_ship_routine/processed/
  - train/*.tab
  - validation/*.tab
  - test/*.tab
- README.txt

Under each directory called train, validation, or test, you will find comma-separated tables with the following columns:
- time [seconds]: The sampling rate is 1 Hz.
- n [1/s]: propeller shaft speed, both rudder propellers have synchronized shaft speeds. Always positive.
- deltal [rad]: azimuth angle for left rudder propeller.
- deltar [rad]: azimuth angle for right rudder propeller.
- Vw [m/s]: Wind speed in intertial frame.
- alpha_x [1]: cos(alpha) where alpha is the wind angle of attack. The angle is already relative to the ship.
- alpha_y [1]: sin(alpha) see above.
- u [m/s]: surge velocity of ship in BODY frame.
- v [m/s]: sway velocity of ship in BODY frame.
- p [rad/s]: roll rate of ship in BODY frame.
- r [rad/s]: yaw rate of ship in BODY frame.
- phi [rad]: roll angle of ship relative to inertial frame.
