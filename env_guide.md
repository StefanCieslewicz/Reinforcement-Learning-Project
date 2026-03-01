# Key design questions\*\*

**🎯 What skill should the agent learn?**

1. Navigate well through a maze
2. Manage risk vs. speed (hole is costly shortcut)
3. Complete the goal (find the diamond)
4. Avoid failure states, such as lava tiles

**👀 What information does the agent need?**
The map layout for sure

**🎮 What actions can the agent take?**
Discrete action space: {UP, DOWN, LEFT, RIGHT}

**🏆 How do we measure success?**
Success is defined by completing the task with minimal cost. What is more we will measure the time spent by counting the number of taken steps. Our targetis to succed in minimum steps while avoiding unneccessary costs. **Failure** is measurred when stepping on a lava tile. The task is succesfull if the agent finds the diamond

Metrics to report, to make a comparison afterwards:

1. Success rate (% episodes finding the diamond)
2. Average steps-to-success (time)
3. Average return (total reward)

**⏰ When should episodes end?**
**Success**: Diamond/s mined
**Failure**: Fall into lava
\*\*Truncation: Step limit reached, such 150 steps.

# Environment specifications

**Map size** = 12x12 grid (2D)
**State space**: $s = (x,y)$ (agent position only)
**Action space**: $A = {UP, DOWN, LEFT, RIGHT}$

**\*Very important**: $|A| * |S| = 576 < 600$, which we comply with\*

Fixed map design:

1. Wall - impassable
2. Empty - normal tile
3. hole - passable, has a penalty cost
4. Lava - terminates the game, if stepped on
5. Diamond - goal tile, it has to be mined
   _Tiles do not change, the map is fixed_
6. Rocky tile - harder to pass compared to a normal tile (slightly higher cost to pass)
