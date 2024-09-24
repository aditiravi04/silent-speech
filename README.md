# Silent Speech Recognition with Jaw Motion Sensors
Aditi Ravindra, Helen Liu, Rachel Jee

# What is this project about?
We want to create an ear-worn system for recognizing unvoiced human commands that is an alternative to voice-based interactions. Voice-based interactions can not only be unreliable in noisy environments and disruptive, but also compromise our own privacy. The core idea behind this is that we will use a twin-IMU set up to track a user's jaw motion & cancel noise (body/head motion), and then break down the word articulation via jaw motion data. It will break it down into syllables, and then phonemes before reconstructing the word using a bi-directional particle filter based on a dictionary of words/commands we store.

# Why is this project exciting for you and your group?
After reading the paper for Paper Review 2, my group and I were very intrigued by the concept of recognizing speech purely via jaw motion sensors. Specifically, the paper gave us a vivid description of the way they “processes jaw motion during word articulation to break each word signal into its constituent syllables, and then further down into phonemes”. 
The paper described Mutlet to be a new alternative to modern issues with voice-based interactions. 


