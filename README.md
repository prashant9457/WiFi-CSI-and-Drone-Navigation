## Part 1: Research & Core understanding 

## Introduction
**Background:** Autonomous indoor drone navigation traditionally relies on **Visual-Inertial Odometry (VIO)**. VIO systems achieve state-estimation by fusing high-frequency kinematic measurements from an Inertial Measurement Unit (IMU) with pixel-level feature tracking from optical cameras. While highly effective in structured, well-lit environments, VIO is fundamentally brittle and prone to catastrophic failure modes under the following tactical conditions:
* **Photometric Degradation:** 
* **Particulate Scattering (Dusty Warehouses):** 
* **Kinematic Motion Blur:** 

**Intuition:** Unlike cameras, Radio Frequency (RF) signals pass through dense airborne dust and operate seamlessly in pitch-black conditions. By capturing how the physical boundaries of a room deform wireless communication signals, CSI allows us to treat ambient Wi-Fi networks as a passive radar system for localized state-estimation.

**Theoretical Deep-Dive & Robotics Critique:** I have done critical review of the state-of-the-art Wi-Fi sensing literature (*SpotFi*, *DeepFi*, and the *Ma et al. Wi-Fi Sensing Survey*). This section explicitly dismantles the physics-based and machine-learning-based assumptions of ground-based CSI models, analyzing exactly why they suffer from systemic degradation when transitioned to a vibrating, tilting quadcopter chassis.

## Literature Deep-Dive: Comprehensive Paper Analysis

### 1. Paper 1: ["A Survey on Wi-Fi Based Contactless Activity Recognition"](http://www-public.imtbs-tsp.eu/~zhang_da/pub/A%20Survey%20on%20Wi-Fi%20based%20Contactless%20Activity%20Recognition_Final.pdf) (Ma et al. | [IEEE Xplore](https://ieeexplore.ieee.org/document/7839615))
This foundational survey establishes the physical mechanisms of RF-based environmental sensing and details the universal signal-processing pipeline. Which built the intution of how things are working and how wireless signals act as a spatial scanner.
![Drone Vibration Noise Profile](./img/intuition.png)
Image shows the intution behind the multipath propagation of signals from the emitter to receiver, which lay down the foundation of the wifi sensing.

## What is Channel State Information (CSI)?

In wireless communications, **Channel State Information (CSI)** is a collection of fine-grained physical layer metrics that describe how a Wi-Fi signal propagates from a transmitter to a receiver. Unlike **RSSI** (Received Signal Strength Indicator)—which is a single, scalar value representing the total aggregated power of a received signal—CSI provides an exhaustive breakdown of the signal's properties across individual frequencies.

Modern Wi-Fi architectures rely on **OFDM (Orthogonal Frequency Division Multiplexing)**. OFDM divides a single wide Wi-Fi channel (e.g., 20 MHz or 40 MHz) into multiple narrow, independent, tightly packed sub-frequencies called **subcarriers**. CSI captures the exact environmental impact on *every single one* of these individual subcarriers.



---
### How CSI Translates Signal Components to Localization

At its core, CSI tracks how the environment alters two primary properties of a Wi-Fi wave across its subcarriers: **Amplitude** and **Phase**.

#### 1. Amplitude Changes ($|H|$) ──► Tracking Obstacles and Shadows
* **The Physics:** Amplitude measures the power or height of the radio wave. As a Wi-Fi wave hits an object, the material absorbs or reflects its energy, casting an "RF shadow".
* **How it helps Localization:** When a person moves or a drone shifts positions, it blocks or opens up specific reflection paths. This causes immediate, localized drops or spikes in amplitude across different subcarriers. By mapping these amplitude patterns, a classifier can identify which room or coordinate zone matches that specific "shadow profile".

#### 2. Phase Changes ($\angle H$) ──► Tracking Distance and Angles
* **The Physics:** Phase measures the time delay or shifts in the wave's cycle relative to when it was sent. 
* **How it helps Localization:** Because radio waves travel at the speed of light, traveling a longer distance takes more time, which rotates the wave's phase. 
  * **Distance (Time of Flight - ToF):** By looking at how the phase shifts across *different subcarrier frequencies*, algorithms can calculate the exact nanosecond travel time, revealing the distance between the device and the router.
  * **Direction (Angle of Arrival - AoA):** By measuring the microscopic phase differences of a single wave hitting Antenna 1 versus Antenna 2 a fraction of a millimeter apart, the system can geometrically calculate the exact angle from which the signal arrived.


* **Amplitude** acts like a camera sensor detecting **shadows and shapes**.
* **Phase** acts like a laser measure detecting **exact distances and angles**.

---

### 4. Why CSI is a "Spatial Scanner"
Because wireless waves propagate via multiple paths simultaneously (bouncing off walls, floors, and dynamic obstacles) before superimposing at the receiver, the CSI matrix acts as a deterministic holographic snapshot of the room. If the physical layout of the room alters—such as an object moving, a drone tilting, or a human walking—the path lengths change, leaving a distinct, readable "dent" across the subcarrier amplitude and phase streams.

* **Core Takeaways**
  * **The Core Mechanism:** The paper details how wireless signals propagate through an indoor space via a direct path—Line of Sight (LOS)—as well as multiple reflection and scattering paths off walls, floors, and ceilings (Multipath Propagation).
  * **Human Body Interaction:** A human body is composed of mostly water, acting as a dielectric material that introduces extra reflection and refraction paths. The receiver records these continuously as distortions in the Wi-Fi signal.
  * **The 4-Step Engineering Pipeline:** This paper provides a highly actionable architectural blueprint by dividing modern RF systems into a standardized workflow:
    1. *Base Signal Selection:* Choosing between Amplitude (stable, maps power loss/shadows) and Phase (ultra-sensitive to millimeter movements but corrupted by hardware clock noise).
    2. *Preprocessing:* Using low-pass/band-pass Butterworth filters to chop off high-frequency hardware static and applying Principal Component Analysis (PCA) to compress noisy subcarriers into dominant spatial components.
    3. *Feature Extraction:* Extracting Time-Domain features (Mean, Standard Deviation) and Frequency-Domain features (using Short-Time Fourier Transforms to build motion spectrograms).
    4. *Classification Models:* Feeding features into standard classifiers like Support Vector Machines (SVM) or Random Forests.
    ![Drone Vibration Noise Profile](./img/pipeline.png)
Framework for Wi-Fi based contactless activity recognition.

---

### 2. Paper 2: ["CSI-Based Fingerprinting for Indoor Localization: A Deep Learning Approach"](https://arxiv.org/pdf/1603.07080) (DeepFi by Wang et al. | [IEEE Xplore](https://ieeexplore.ieee.org/document/7442544))
DeepFi represents the data-driven "Fingerprinting" paradigm, completely bypassing manual physics calculations by using deep neural networks to learn spatial patterns.
![Drone Vibration Noise Profile](./img/Architecture.png)

* **Core Takeaways & Attention-Grabbing Insights:**
  * **The 90-Dimensional Feature Vector:** DeepFi leverages a commercial Intel 5300 Network Interface Card (NIC) equipped with 3 physical antennas. The custom drivers expose 30 Orthogonal Frequency Division Multiplexing (OFDM) subcarriers per antenna. DeepFi multi-plexes these into a raw $3 \times 30 = 90$ dimensional matrix of subcarrier amplitudes for every single packet.
  * **The Two-Phase Architecture:** * *Offline Training:* Collecting the 90-dimensional amplitude fingerprints at known grid coordinates across a room.
    * *Online Localization:* Matching real-time incoming CSI amplitudes against the learned radio map using a probabilistic Radial Basis Function (RBF) kernel to output a location coordinate.
  * **Unsupervised RBM Stack:** Rather than standard convolutional or feed-forward networks, DeepFi utilizes a deep network composed of **Restricted Boltzmann Machines (RBMs) with 4 hidden layers**. It implements a **Greedy Layer-by-Layer Unsupervised Learning** algorithm, which optimizes weights one layer at a time to vastly reduce the onboard compute power required to train the spatial fingerprints.

---

### 3. Paper 3: ["SpotFi: Decimeter Level Localization Using WiFi"](https://web.stanford.edu/~skatti/pubs/sigcomm15-spotfi.pdf) (Kotaru et al. | [ACM Digital Library](https://dl.acm.org/doi/10.1145/2785956.2787487))
SpotFi is an absolute masterpiece of pure geometry and physics. It completely rejects machine learning and neural networks, achieving centimeter-level localization using advanced array signal processing.

* **Core Takeaways & Attention-Grabbing Insights:**
  * **The Virtual Antenna Array:** Standard off-the-shelf Wi-Fi cards only have 3 physical antennas, meaning they can mathematically only resolve 2 distinct multipath waves before getting blinded. SpotFi brilliantly stiches the phase and amplitude data across all 3 physical antennas *and* 30 subcarrier frequencies together. This tricks the system into creating a massive "Virtual Sensor Array," giving it the high-resolution power to isolate dozens of bouncing echoes simultaneously.
  * **The 2-D MUSIC Super-Resolution Algorithm:** SpotFi passes this virtual array matrix into a modified **Multiple Signal Classification (MUSIC)** algorithm. This allows the system to compute two values for every incoming radio wave simultaneously:
    1. *AoA (Angle of Arrival):* The exact angle relative to the antenna array from which the wave arrived.
    2. *ToF (Time of Flight):* The precise nanosecond-scale time delay the wave took to travel from transmitter to receiver.
  * **Direct Path Identification:** In a room full of reflections, the wave bouncing off a metal wall might look much stronger than the true path. SpotFi implements a filter that isolates the true straight-line path between the drone and the router by identifying the path with the **absolute shortest Time of Flight (ToF)**, allowing for precise geometric triangulation.


## Why Existing CSI Localization Methods Break Down on a Drone

*A common assumption across the Wi-Fi sensing survey, DeepFi, and SpotFi is that the wireless infrastructure and sensing platform remain stationary while the environment changes. In the datasets used by these works, CSI is typically collected using fixed access points and receivers placed at known locations with stable antenna orientations. Under these conditions, variations in CSI primarily reflect environmental factors such as human motion, multipath reflections, or changes in object positions.*

A drone-mounted receiver violates this assumption. Unlike a static laptop or embedded device, a drone continuously changes its position, altitude, velocity, and orientation during flight. As a result, every multipath component between the transmitter and receiver becomes time-varying. Changes in CSI are no longer caused solely by the environment; they are also caused by the motion of the sensing platform itself. This makes it difficult to separate environmental information from platform-induced artifacts.

For fingerprinting-based approaches such as DeepFi, the problem is that CSI fingerprints collected during training are highly dependent on receiver position and antenna orientation. Small changes in altitude, roll, pitch, or yaw can significantly alter the observed channel response, causing the measured CSI to deviate from the fingerprints stored in the database. Consequently, the learned model experiences a distribution shift between training and deployment conditions, reducing localization accuracy.

For geometry-based approaches such as SpotFi, the issue is even more severe. SpotFi estimates Angle of Arrival (AoA) and Time of Flight (ToF) using precise phase relationships across antennas and subcarriers. Drone vibrations generated by motors and propellers introduce phase disturbances at the antenna level, while vehicle motion changes path lengths during packet capture. These effects corrupt the phase measurements required for accurate AoA and ToF estimation, degrading the performance of the MUSIC-based localization pipeline.

In addition, propellers operating near the antennas can periodically modulate the received RF signal and introduce artificial Doppler-like effects. Antenna tilting during flight can also create polarization mismatches with the transmitter, causing signal fluctuations that are unrelated to the surrounding environment. Therefore, many of the CSI variations observed on a drone originate from the flight platform itself rather than the environment being sensed.

In summary, existing CSI localization systems assume that the receiver acts as a stable observer of the wireless channel. On a drone, the receiver becomes part of the channel dynamics. The resulting motion-induced phase noise, orientation changes, vibration effects, and continuously evolving multipath geometry violate the core assumptions underlying current CSI fingerprinting and localization methods, making direct deployment on aerial platforms significantly more challenging.

---

## Part 2: Dataset — Widar 3.0 BVP

### What is BVP and Why Does it Matter?

Rather than classifying raw CSI measurements, Widar 3.0 introduces a domain-independent feature called the **Body-coordinate Velocity Profile (BVP)**. The core insight is simple: *a human gesture produces a unique pattern of velocities regardless of where in the room the person stands or which direction they face*. The BVP captures exactly that — a 2-D snapshot of how much signal energy is moving at each combination of X-velocity and Y-velocity at a given instant.

Each `.mat` file stores a **`velocity_spectrum_ro`** tensor of shape **(20 × 20 × T)**:

| Axis | Meaning | Range |
|---|---|---|
| Rows (20) | Velocity along X | −2 m/s → +2 m/s, step 0.2 m/s |
| Cols (20) | Velocity along Y | −2 m/s → +2 m/s, step 0.2 m/s |
| T (frames) | Time | ~100 ms per frame, median T ≈ 17 |

Because velocity is body-centred, the BVP is largely invariant to receiver placement and room geometry — making it a strong foundation for cross-environment gesture recognition.

---

### BVP Visualization — Push & Pull (Gesture 1)

The three panels below collapse the time axis in different ways to reveal the gesture's velocity fingerprint:

![BVP Projections](./img/bvp_projections.png)

| Panel | What it shows |
|---|---|
| **Middle Frame** | A snapshot at the midpoint of the gesture |
| **Max Projection** | Every velocity cell the body ever activated — the full gesture footprint |
| **Mean Projection** | Where energy was concentrated most consistently across all frames |

The animation below plays back each time-frame at 10 fps, showing how the velocity blob moves through the grid during the gesture:

![BVP Animation](./img/bvp_animation.gif)

Run the visualizer yourself to explore any gesture:

```bash
python code/visualize_bvp.py   # change GESTURE_ID (1–10) at the bottom of the script
```

---

### Dataset Structure

```
code/data/BVP/
└── {Date}-VS/           ← 14 recording sessions  (20181109 → 20181211)
    └── 6-link/          ← 6-receiver link topology
        └── user{N}/     ← 17 subjects
            └── *.mat    ← one BVP tensor per gesture instance
```

| Dimension | Count | Values |
|---|---|---|
| Subjects | 17 | user1 – user17 |
| Gesture classes | 10 | Push&Pull, Sweep, Clap, Slide, Circle ×2, Triangle, Zigzag, N, Random |
| Locations per room | 8 | grid positions |
| Orientations | 5 | cardinal + diagonal directions |
| Repetitions | up to 20 | per condition |
| **Total .mat files** | **43 658** | |

> **Note:** One file (`user15-5-4-5-2-…-L0.mat`) is a known truncated write on disk and is skipped automatically by the loader.

---

### Filename Convention

Files appear in three formats depending on the recording session. The first five fields are always identical:

```
user{U} - {gesture} - {location} - {orientation} - {repetition} - …
```

| Format | Example |
|---|---|
| Parameter | `user1-1-1-1-1-1-1e-07-100-20-100000-L0.mat` |
| Date | `user2-1-1-1-1-20181208.mat` |
| Parameter + Date | `user1-1-1-1-1-1-1e-07-100-20-100000-L0-20181121.mat` |

The trailing parameters in the longer formats encode the BVP solver configuration:

| Token | Meaning | Value |
|---|---|---|
| `p6` | Algorithm flag | `1` |
| `p7` | Regularisation weight (η) | `1e-07` |
| `p8` | Max solver iterations | `100` |
| `p9` | Velocity grid resolution | `20` |
| `p10` | Scale factor | `100000` |
| `p11` | Norm type | `L0` |

---

### Class Distribution

Gestures 1–6 span all 14 sessions and are well-represented; Gestures 7–9 appear in fewer sessions; Gesture 10 is the smallest class. Any classifier must account for this imbalance via **class weighting** or **stratified sampling**.

| Gesture | Name | Samples |
|---|---|---|
| 1 | Push & Pull | 6 547 |
| 2 | Sweep | 6 424 |
| 3 | Clap | 6 421 |
| 4 | Slide | 6 300 |
| 5 | Draw Circle (CW) | 6 175 |
| 6 | Draw Circle (CCW) | 6 041 |
| 7 | Draw Triangle | 1 750 |
| 8 | Draw Zigzag | 1 750 |
| 9 | Draw N | 1 750 |
| 10 | Random | 500 |

