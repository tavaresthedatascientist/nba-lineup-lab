# OKC lineup examples

Generated with `python -m lineup_lab optimize` on the model trained through the end of the 2025-26 regular season. Opponent fives are each team's most-used starting group over its last 15 games.

```

Oklahoma City Thunder vs San Antonio Spurs (home)
Core (always starts when available): Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren
Opponent starting five: Devin Vassell, Victor Wembanyama, Stephon Castle, Julian Champagnie, De'Aaron Fox
Most-used recent five: Shai Gilgeous-Alexander, Chet Holmgren, Ajay Mitchell, Luguentz Dort, Isaiah Hartenstein -> 61.3%

Top lineups (28 valid lineups evaluated):
  1. 68.8%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Isaiah Hartenstein, Alex Caruso
  2. 68.8%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Isaiah Hartenstein, Isaiah Joe
  3. 68.7%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Isaiah Hartenstein, Jaylin Williams

Oklahoma City Thunder vs Denver Nuggets (away)
Core (always starts when available): Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren
Opponent starting five: Jamal Murray, Nikola Jokić, Christian Braun, Cameron Johnson, Tim Hardaway Jr.
Most-used recent five: Shai Gilgeous-Alexander, Chet Holmgren, Ajay Mitchell, Luguentz Dort, Isaiah Hartenstein -> 67.5%

Top lineups (28 valid lineups evaluated):
  1. 78.4%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Ajay Mitchell, Isaiah Hartenstein
  2. 77.0%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Isaiah Hartenstein, Isaiah Joe
  3. 74.6%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Ajay Mitchell, Isaiah Joe

Oklahoma City Thunder vs Houston Rockets (home)
Core (always starts when available): Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren
Opponent starting five: Amen Thompson, Jabari Smith Jr., Alperen Sengun, Kevin Durant, Tari Eason
Most-used recent five: Shai Gilgeous-Alexander, Chet Holmgren, Ajay Mitchell, Luguentz Dort, Isaiah Hartenstein -> 67.4%

Top lineups (28 valid lineups evaluated):
  1. 80.6%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Ajay Mitchell, Isaiah Hartenstein
  2. 78.9%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Isaiah Hartenstein, Isaiah Joe
  3. 78.8%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Ajay Mitchell, Isaiah Joe

Oklahoma City Thunder vs Minnesota Timberwolves (away)
Core (always starts when available): Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren
Opponent starting five: Jaden McDaniels, Julius Randle, Naz Reid, Rudy Gobert, Anthony Edwards
Most-used recent five: Shai Gilgeous-Alexander, Chet Holmgren, Ajay Mitchell, Luguentz Dort, Isaiah Hartenstein -> 69.6%

Top lineups (28 valid lineups evaluated):
  1. 82.8%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Ajay Mitchell, Isaiah Hartenstein
  2. 82.3%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Ajay Mitchell, Isaiah Joe
  3. 82.0%  Shai Gilgeous-Alexander, Jalen Williams, Chet Holmgren, Ajay Mitchell, Alex Caruso

$ python -m lineup_lab optimize --team OKC --opp SAS --home --out "Jalen Williams" --top 3

Oklahoma City Thunder vs San Antonio Spurs (home)
Core (always starts when available): Shai Gilgeous-Alexander, Chet Holmgren
Opponent starting five: Devin Vassell, Victor Wembanyama, Stephon Castle, Julian Champagnie, De'Aaron Fox
Most-used recent five: Shai Gilgeous-Alexander, Chet Holmgren, Ajay Mitchell, Luguentz Dort, Isaiah Hartenstein -> 61.3%

Top lineups (84 valid lineups evaluated):
  1. 67.0%  Shai Gilgeous-Alexander, Chet Holmgren, Ajay Mitchell, Isaiah Hartenstein, Isaiah Joe
  2. 66.3%  Shai Gilgeous-Alexander, Chet Holmgren, Ajay Mitchell, Isaiah Hartenstein, Alex Caruso
  3. 66.3%  Shai Gilgeous-Alexander, Chet Holmgren, Ajay Mitchell, Isaiah Hartenstein, Jaylin Williams
```
