# Play classifier report

- top-1 @ 0.5 s: 0.8369 raw, 0.8564 with deck inference (564 plays)
- top-1 @ 2 s: 0.8582 raw, 0.8883 with deck inference (564 plays)
- no_play precision 0.970, recall 0.946

## Top confusions (true -> predicted)

- no_play -> the_log: 7
- no_play -> goblin_drill: 5
- no_play -> royal_delivery: 5
- no_play -> barbarian_barrel: 5
- the_log -> no_play: 4
- fireball -> no_play: 4
- no_play -> giant_snowball: 4
- no_play -> fireball: 4
- goblin_barrel -> no_play: 4
- no_play -> tornado: 3
- fireball -> barbarian_barrel: 3
- giant_snowball -> no_play: 3
- no_play -> goblin_cage: 3
- no_play -> goblins: 2
- no_play -> goblin_curse: 2

## Worst cards

- monk: 0.000
- rocket: 0.000
- the_log: 0.231
- fireball: 0.294
- x_bow: 0.333
- giant_snowball: 0.385
- goblin_barrel: 0.500
- graveyard: 0.500
- barbarian_barrel: 0.556
- ice_wizard: 0.643
- archer_queen: 0.667
- miner: 0.667
- heal_spirit: 0.700
- freeze: 0.750
- royal_giant: 0.750
- cannon_cart: 0.800
- mini_pekka: 0.800
- mortar: 0.800
- tornado: 0.800
- balloon: 0.833

## Proposal recall

recall by kind: {'Troop': '70/106 = 0.660', 'Building': '16/22 = 0.727', 'Spell': '4/43 = 0.093'}
overall recall: 0.5263157894736842 | proposals: 1052 phantom: 962
