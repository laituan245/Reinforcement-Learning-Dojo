## Cross-entropy on CartPole

The steps of the cross-entropy method are:
1. Play *N* number of episodes using the current model and environment.
2. Calculate the total reward for every episode and decide on a reward boundary.
3. Throw away all episodes with a total reward below the boundary (i.e., keep episodes with high final total rewards).
4. Train on the remaining  "elite" episodes using observations as the input and issued actions as the desired output.
5. Repeat from step 1 until we become satisfied with the result.

Demo:
<p align="center">
  <img src="https://github.com/laituan245/Reinforcement-Learning-Dojo/blob/master/cross_entropy/demo.gif">
</p>
