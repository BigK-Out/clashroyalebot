# Play classifier report

- top-1 @ 0.5 s: 0.8732 raw, 0.8875 with deck inference (489 plays)
- top-1 @ 2 s: 0.8937 raw, 0.9100 with deck inference (489 plays)
- no_play precision 0.980, recall 0.944

## Top confusions (true -> predicted)

- no_play -> arrows: 6
- no_play -> royal_delivery: 5
- no_play -> giant_snowball: 5
- no_play -> fireball: 4
- no_play -> ice_wizard: 3
- the_log -> no_play: 3
- goblin_barrel -> royal_delivery: 3
- fireball -> no_play: 3
- no_play -> goblins: 2
- no_play -> goblin_barrel: 2
- fireball -> royal_delivery: 2
- no_play -> goblin_drill: 2
- giant_snowball -> no_play: 2
- miner -> giant: 1
- no_play -> cannon_cart: 1

## Worst cards

- barbarian_barrel: 0.000
- rocket: 0.000
- goblin_barrel: 0.143
- the_log: 0.143
- fireball: 0.250
- giant_snowball: 0.429
- ice_wizard: 0.545
- miner: 0.667
- x_bow: 0.667
- three_musketeers: 0.750
- cannon_cart: 0.800
- heal_spirit: 0.800
- mini_pekka: 0.800
- mortar: 0.800
- phoenix: 0.800
- tombstone: 0.800
- tornado: 0.800
- bomb_tower: 0.833
- goblin_demolisher: 0.833
- void: 0.833

## Proposal recall

recall by kind: {'Troop': '70/106 = 0.660', 'Building': '16/22 = 0.727', 'Spell': '4/43 = 0.093'}
overall recall: 0.5263157894736842 | proposals: 1052 phantom: 962
