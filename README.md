# Silent Speech Recognition with Jaw Motion Sensors
Group members: Aditi Ravindra, Helen Liu, Rachel Jee

# What is this project about?
We want to create an ear-worn system for recognizing unvoiced human commands that is an alternative to voice-based interactions. Voice-based interactions can not only be unreliable in noisy environments and disruptive, but also compromise our own privacy. The core idea behind this is that we will use a twin-IMU set up to track a user's jaw motion & cancel noise (body/head motion), and then break down the word articulation via jaw motion data. It will break it down into syllables, and then phonemes before reconstructing the word using a bi-directional particle filter based on a dictionary of words/commands we store.


