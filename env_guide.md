# Key design questions\*\*

**🎯 What skill should the agent learn?**

1. Navigate well through a maze
2. Manage risk vs. speed (lava is costly shortcut)
3. Complete the goal (mine the diamond)
4. Avoid failure states, such as zombie tiles

**👀 What information does the agent need?**
The map layout for sure

**🎮 What actions can the agent take?**
Discrete action space: {FORWARD, BACKWARDS, LEFT, RIGHT, MINE}

**🏆 How do we measure success?**
Success is defined by completing the task with minimal cost. What is more we will measire the time spent by counting the number of taken steps. Our targetis to succed in minimum steps while avoiding unneccessary costs. **Failure** is measurred when stepping on a zombie tile.

Metrics to report, to make a comparison afterwards:

1. Success rate (% episodes mining diamond)
2. Average steps-to-success (time)
3. Average return (total reward)

**⏰ When should episodes end?**
**Success**: Diamond mined
**Failure**: Zombie stepped on
\*\*Truncation: Step limit reached, such 150 steps.

# Environment specifications

**Map size** = 10x10 grid
**State space**: $s = (x,y)$ (agent position only)
**Action space**: $A = {FORWARD, BACKWARDS, LEFT, RIGHT, MINE}$

**\*Very important**: $|A| * |S| < 600$, which we comply with\*

Fixed map design:

1. Wall - impassable
2. Empty - normal tile
3. Lava - passable, has a penalty cost
4. Zombie - terminates the game, if stepped on
5. Diamond - goal tile, it has to be mined
   _Tiles do not change, the map is fixed_
