Simulation - Groudh Truth - Sense Data (parameters, can add in noise) - Prob/Reconstruction (Kalman filter) - Compare (Accuracy -> Grouping)

most: related work

**what’s contract tracing**

**related work - potential risks**

**data - real data / simulation**

Sensing data:

- All of the attackers' stationary devices exchanges the information with the walkers every 1 min and at **exactly the same Ti**.
- The device id doesn't change for now.

Pre-condition:

- The walkers' ids are updated at different time.

- The walkers' ids are updated every 15 min. 

Goal:

- Group ids

Ground Truth:

- The actual path without noise.

Parameter:

- how many minustes we gather data
- noise
- how oftern change id



The sub-path is now generated from 15 points.

- Use Kalman filter to hypothesis the position of next state.
- Get the probability of the each record at the next-time
- Continue the process until the last time Ti



TODO:

add in walker collecting data

add noise to device

