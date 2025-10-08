# State Space Models - SSM

This workspace contains the research and implementation of dynamic state-space models for predicting football match outcomes. The primary goal is to move beyond static models by capturing the time-varying nature of team strengths. The models implemented here treat team attack and defense abilities as latent variables that evolve from week to week according to a first-order autoregressive (AR(1)) process.

This work is heavily inspired by the methodologies presented in the academic paper by Koopman & Lit (2012), 

"A Dynamic Bivariate Poisson Model for Analysing and Forecasting Match Results in the English Premier League".




# Look at Rangers' Draw and the Aftermath

![alt text](https://github.com/JamesZor/models_julia/blob/7-feat-basic-state-space-models/rangers_falkirk_performance.png)

When Rangers, a titan of Scottish football, drew 1-1 with newly-promoted Falkirk, the result sent shockwaves that ended with the manager's dismissal.
On the surface, it looked like a classic "giant-killing" upset.
But was it really a surprise? I had a hunch that the data might tell a different story.
Using a dynamic state-space model, I tracked the underlying form of both clubs over the last five seasons to see what was really going on behind the scenes.

## The Tale of Two Clubs
The model rates each team's attacking and defensive strength over time.
When plotted, the trajectories of Rangers and Falkirk reveal a stark narrative that the league table alone doesn't show.
What we're seeing here is a story of two clubs heading in opposite directions:Rangers' Slow Puncture: The orange line shows a clear and steady decline in both attacking and defensive prowess over several years.
This isn't a recent dip in form; it's a long-term erosion of performance.
The manager, hired at the start of the 25/26 season (the red dotted line), walked into a team already on a significant downward slope and couldn't pull them out of the nosedive.
Falkirk's Steady Climb: The blue line, meanwhile, is the picture of progress.
Falkirk has been consistently improving season after season, closing the gap on the teams above them.
The 1-1 draw wasn't a fluke.
It was the inevitable intersection point of a fading giant and a rising contender.
## Spotting What the Bookies Missed
If the trend was so clear, did the betting market see it? Not quite.
My model's odds for the match looked quite different from the bookmaker's.
### Outcome 
The model correctly saw that Rangers were being significantly overvalued.


| Outcome | Model | Bookmaker | Result |
|---------|-------|-----------|--------|
| Falkirk | 4.29  | 4.73      | -      |
| Draw    | 4.52  | 4.20      | x      |
| Rangers | 2.03  | 1.62      | -      |


It picked up on the long-term decline and assessed their chances of winning as being much lower than the bookies thought, flagging the 1.62 odds as extremely poor value.

