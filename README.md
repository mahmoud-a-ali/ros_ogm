# ogm: occupancy grid map

#### `ogm.py` is a ros node which subscribes to two topics to produce the occupancy grid map:
- `/front/scan`: Laserscan topic with a ros msg of type `sensor_msgs/LaserScan`.
- `/ground_truth/state`: Localization topic with a ros msg of type `nav_msgs/Odometry`.

#### to have it running,
- first, run real or simulated robot with a 2d laserscanner, make sure you have both localization and laserscan topic is published.
- run `ogm.py` script: ``` python3 ogm.py```. you can also put this node in a ros package and use `rosrun` or `roslaunch` command to run it.

#### video: https://youtu.be/Ih2ayEjO228
