//! Decision making: the `Policy` trait and a rule-based Hog 2.6 bot.

mod cards;
pub mod enemy;

pub use cards::{ROLES, Card, Cards, cards};

use std::time::Duration;

use state::{GameState, Lane, MY_FIRST_ROW_HINT};

/// Why an action was chosen (logged, and used by the policy's own timers).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Why {
    /// Spell on a group of enemies.
    Swarm,
    /// Troop/building against enemies on my side of `lane`.
    Defend(Lane),
    /// Hog Rider push.
    Push(Lane),
    /// Ice Spirit right behind the Hog Rider.
    FollowUp,
    /// Fireball on a tower: finishing a low one, or at full elixir with nothing else to play.
    TowerSpell,
    /// Elixir about to overflow.
    Leak,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Deploy {
        slot: usize,
        card: String,
        col: u32,
        row: u32,
        /// Target is on the enemy half (spells).
        enemy_half: bool,
        why: Why,
    },
}

pub trait Policy {
    fn decide(&mut self, state: &GameState) -> Option<Action>;

    /// Called after an action was executed, so the policy can track timers.
    fn on_action(&mut self, _action: &Action, _state: &GameState) {}
}

/// Elixir cost of a card (from `assets/cards.json`); `None` for unknown cards and Mirror.
pub fn cost(card: &str) -> Option<u8> {
    cards().get(card)?.elixir
}

/// Bridge tile on my side.
fn bridge(lane: Lane) -> (u32, u32) {
    match lane {
        Lane::Left => (3, 17),
        Lane::Right => (14, 17),
    }
}

/// Enemy princess tower center (spell target).
fn enemy_tower(lane: Lane) -> (u32, u32) {
    match lane {
        Lane::Left => (3, 6),
        Lane::Right => (14, 6),
    }
}

/// Behind my princess tower (ranged support).
fn behind_tower(lane: Lane) -> (u32, u32) {
    match lane {
        Lane::Left => (3, 26),
        Lane::Right => (14, 26),
    }
}

/// Center tile slightly toward `lane`: buildings here pull lane attackers between both towers.
fn center_pull(lane: Lane) -> (u32, u32) {
    match lane {
        Lane::Left => (8, 21),
        Lane::Right => (9, 21),
    }
}

/// Clamp a troop target onto my half.
fn my_side(col: u32, row: u32) -> (u32, u32) {
    (col.min(17), row.clamp(MY_FIRST_ROW_HINT, 31))
}

/// Largest group of units within `radius` tiles of one of them: (center col, center row, count).
fn densest(units: &[(u32, u32)], radius: f32) -> Option<(u32, u32, usize)> {
    units
        .iter()
        .map(|&(c, r)| {
            let near: Vec<_> = units
                .iter()
                .filter(|&&(c2, r2)| ((c as f32 - c2 as f32).powi(2) + (r as f32 - r2 as f32).powi(2)).sqrt() <= radius)
                .collect();
            let n = near.len();
            let cc = near.iter().map(|u| u.0 as f32).sum::<f32>() / n as f32;
            let rr = near.iter().map(|u| u.1 as f32).sum::<f32>() / n as f32;
            (cc.round() as u32, rr.round() as u32, n)
        })
        .max_by_key(|g| g.2)
}

/// Hog 2.6 with enemy awareness (unit tags, no unit types):
/// 1. defend: enemies on my side of a lane → spell a swarm, else a defensive card for that lane;
/// 2. push: Hog Rider (+ Ice Spirit) when my side is clear, into the lane just defended or the
///    weaker enemy tower;
/// 3. fallbacks: play a cheap card before elixir overflows (Fireball the tower only if nothing else fits).
pub struct HogCycle {
    /// Elixir at which Hog Rider goes in.
    pub hog_at: u8,
    /// Elixir at which a card is played to avoid leaking.
    pub leak_at: u8,
    /// Push lane when no recent defense suggests one.
    pub default_lane: Lane,
    /// Elixir is free for the first seconds; don't spend it all before anything happens.
    pub opening_wait: Duration,
    last_hog: Option<(Duration, Lane)>,
    last_defense: [Option<Duration>; 2],
    last_defense_lane: Option<(Duration, Lane)>,
    /// Defensive spending per lane: (battle time, elixir).
    spent: [Vec<(Duration, u8)>; 2],
}

impl Default for HogCycle {
    fn default() -> Self {
        Self {
            hog_at: 6,
            leak_at: 9,
            default_lane: Lane::Right,
            opening_wait: Duration::from_secs(8),
            last_hog: None,
            last_defense: [None; 2],
            last_defense_lane: None,
            spent: [Vec::new(), Vec::new()],
        }
    }
}

const FOLLOW_UP: Duration = Duration::from_millis(1500);
/// One defensive card per lane per this long (unless the threat grows).
const DEFENSE_COOLDOWN: Duration = Duration::from_millis(3000);
/// Counter-push into the lane defended within this long.
const COUNTER_WINDOW: Duration = Duration::from_secs(8);
const SWARM_RADIUS: f32 = 2.5;
const SWARM_MIN: usize = 3;
/// Units that die to The Log: two of these together already justify a spell.
const SWARM_KINDS: [&str; 4] = ["skeletons", "goblins", "barbarians", "archers"];
/// Defensive elixir per lane within this window is capped by `defense_budget`.
const BUDGET_WINDOW: Duration = Duration::from_secs(4);
/// Counter-push Hog Rider needs only this much elixir right after a defense.
const COUNTER_HOG_AT: u8 = 4;
/// Double elixir starts at 2:00.
const DOUBLE_ELIXIR: Duration = Duration::from_secs(120);
/// Enemy elixir at or below this (from recognized plays): Hog Rider goes in to punish.
const PUNISH_AT: f32 = 2.5;
/// Recognized enemy plays this recent still describe what is on the field.
const RECENT_PLAY: Duration = Duration::from_secs(8);
/// Enemy tower at or below this HP fraction: Fireball finishes it (2-3 Fireballs).
const FIREBALL_FINISH: f32 = 0.12;
/// Their side's back rows: a tank or pump placed here is slow to arrive (punish the other lane).
const ENEMY_BACK_ROW: u32 = 8;
/// An enemy tower this much lower (HP fraction) than the other becomes the push lane.
const WEAKER_TOWER: f32 = 0.15;
/// Enemy cards The Log kills or pushes away (Hog 2.6 guide: pre-Log their Hog defenders).
const LOG_KILLS: [&str; 13] = [
    "skeletons", "goblins", "spear_goblins", "skeleton_army", "goblin_gang", "guards", "princess",
    "dart_goblin", "wall_breakers", "firecracker", "bomber", "rascals", "fire_spirit",
];

fn enemy_deck(s: &GameState) -> enemy::EnemyDeck {
    let mut d = enemy::EnemyDeck::default();
    for p in &s.enemy_plays {
        d.on_play(&p.card);
    }
    d
}

fn recent_play<'a>(s: &'a GameState, pred: impl Fn(&cards::Card) -> bool) -> Option<&'a state::EnemyPlay> {
    s.enemy_plays
        .iter()
        .rev()
        .take_while(|p| s.battle_time.saturating_sub(p.t) < RECENT_PLAY)
        .find(|p| cards().get(&p.card).is_some_and(&pred))
}

/// Their Hog counters likely in hand that a pre-Log cannot clear.
fn blocking_hog_counters(s: &GameState) -> bool {
    let d = enemy_deck(s);
    let log_ready = s.slot_of("the_log").is_some() && s.elixir >= 6;
    d.hog_counters()
        .iter()
        .any(|c| d.in_hand(c) == enemy::InHand::Likely && !(log_ready && LOG_KILLS.contains(c)))
}

/// Enemy towers destroyed minus mine (from the HP bars).
fn tower_lead(s: &GameState) -> i32 {
    s.towers.map_or(0, |t| t.enemy.iter().filter(|h| h.is_none()).count() as i32 - t.mine.iter().filter(|h| h.is_none()).count() as i32)
}

/// Elixir allowed on defense in one lane per window: about what the enemy spent, plus 1.
fn defense_budget(push_value: f32) -> u8 {
    (push_value + 1.0).clamp(0.0, 7.0) as u8
}

/// Pushes worth this much or less are left to the towers (Skeletons, Ice Spirit, few Goblins).
const TOWER_HANDLES: f32 = 2.0;

/// Enemy card cost for a classified unit type (unit types are card slugs), and whether the
/// card spawns several units (then a whole group of that type counts as one card).
fn enemy_card(kind: &str) -> Option<(f32, bool)> {
    let c = cards().get(kind)?;
    Some((c.elixir? as f32, c.swarm))
}

/// Estimated elixir value of a group of enemy units (one lane).
///
/// Known swarm types count once per group (a 15-unit Skeleton Army is one card; more than
/// four skeletons together = Skeleton Army, 3 elixir). Unknown units are grouped by
/// proximity: a lone unit ~3.5, a clump of 2+ ~4 (one swarm card or a pair).
pub fn push_value(threats: &[((u32, u32), Option<&str>)]) -> f32 {
    let mut value = 0.0;
    let mut swarm_groups: Vec<(&str, Vec<(u32, u32)>)> = Vec::new();
    let mut unknown: Vec<(u32, u32)> = Vec::new();
    for &(t, k) in threats {
        match k.and_then(|k| enemy_card(k).map(|c| (k, c))) {
            Some((k, (_, true))) => match swarm_groups.iter_mut().find(|(gk, g)| *gk == k && g.iter().any(|&u| near(u, t, 3.0))) {
                Some((_, g)) => g.push(t),
                None => swarm_groups.push((k, vec![t])),
            },
            Some((_, (c, false))) => value += c,
            None => unknown.push(t),
        }
    }
    for (k, g) in &swarm_groups {
        let (c, _) = enemy_card(k).unwrap();
        value += if *k == "skeletons" && g.len() > 4 { 3.0 } else { c };
    }
    // Unknown units: cluster greedily.
    while let Some(first) = unknown.pop() {
        let mut n = 1;
        unknown.retain(|&u| {
            let close = near(u, first, 2.5);
            n += close as usize;
            !close
        });
        value += if n == 1 { 3.5 } else { 4.0 };
    }
    value
}

fn near(a: (u32, u32), b: (u32, u32), r: f32) -> bool {
    ((a.0 as f32 - b.0 as f32).powi(2) + (a.1 as f32 - b.1 as f32).powi(2)).sqrt() <= r
}

fn lane_idx(l: Lane) -> usize {
    match l {
        Lane::Left => 0,
        Lane::Right => 1,
    }
}

impl HogCycle {
    fn deploy(s: &GameState, card: &str, (col, row): (u32, u32), enemy_half: bool, why: Why) -> Option<Action> {
        let slot = s.slot_of(card)?;
        (s.elixir >= cost(card)?).then(|| Action::Deploy { slot, card: card.into(), col, row, enemy_half, why })
    }

    /// Lane to push without a counter-push reason: the clearly weaker enemy princess tower,
    /// else `default_lane`.
    fn push_lane(&self, s: &GameState) -> Lane {
        match s.towers.map(|t| t.enemy) {
            Some([Some(l), Some(r)]) if r - l >= WEAKER_TOWER => Lane::Left,
            Some([Some(l), Some(r)]) if l - r >= WEAKER_TOWER => Lane::Right,
            _ => self.default_lane,
        }
    }

    fn spent_recently(&self, s: &GameState, lane: Lane) -> u8 {
        self.spent[lane_idx(lane)]
            .iter()
            .filter(|(t, _)| s.battle_time.saturating_sub(*t) < BUDGET_WINDOW)
            .map(|(_, c)| c)
            .sum()
    }

    fn defend(&self, s: &GameState, lane: Lane) -> Option<Action> {
        let threats = s.threats(lane);
        let kinds = s.threat_kinds(lane);
        // Most advanced attacker (closest to my towers).
        let &(fc, fr) = threats.iter().max_by_key(|t| t.1)?;
        let value = push_value(&kinds);
        // Flying attackers (unit classifier or a recognized play in this lane): only
        // air-targeting cards help; The Log, Cannon, Ice Golem and Skeletons do not.
        let air = kinds.iter().any(|(_, k)| k.is_some_and(enemy::is_air))
            || s.enemy_plays.iter().any(|p| {
                s.battle_time.saturating_sub(p.t) < RECENT_PLAY && Lane::of_col(p.tile.0) == lane && enemy::is_air(&p.card)
            });
        // Cheap pushes: the towers win that trade for free.
        if value <= TOWER_HANDLES {
            return None;
        }

        // Swarm: 3+ units together, or 2+ known swarm units together; only spell it when the
        // swarm is worth more than the spell (positive trade).
        let swarmy: Vec<(u32, u32)> = kinds
            .iter()
            .filter(|(_, k)| k.is_some_and(|k| SWARM_KINDS.contains(&k)))
            .map(|(t, _)| *t)
            .collect();
        let group = densest(&threats, SWARM_RADIUS)
            .filter(|g| g.2 >= SWARM_MIN)
            .or_else(|| densest(&swarmy, SWARM_RADIUS).filter(|g| g.2 >= 2));
        if let Some((gc, gr, _)) = group {
            let in_group: Vec<_> = kinds.iter().copied().filter(|(t, _)| near(*t, (gc, gr), SWARM_RADIUS)).collect();
            let group_value = push_value(&in_group);
            // The Log rolls up the arena from where it lands: drop it just behind the group.
            let d = enemy_deck(s);
            let barrel_coming = d.in_hand("goblin_barrel") == enemy::InHand::Likely && group_value < 5.0;
            if group_value >= 3.0
                && !air
                && !barrel_coming
                && let Some(a) = Self::deploy(s, "the_log", my_side(gc, gr + 2), false, Why::Swarm)
            {
                return Some(a);
            }
            if group_value >= 5.0
                && let Some(a) = Self::deploy(s, "fireball", (gc, gr), gr < MY_FIRST_ROW_HINT, Why::Swarm)
            {
                return Some(a);
            }
        }

        // Troops: one per lane per cooldown, and within the lane's elixir budget.
        let recent = self.last_defense[lane_idx(lane)].is_some_and(|t| s.battle_time.saturating_sub(t) < DEFENSE_COOLDOWN);
        if recent {
            return None;
        }
        let left = defense_budget(value).saturating_sub(self.spent_recently(s, lane));
        // Toward the center from the attacker, so it gets pulled between both towers.
        let toward_center = if lane == Lane::Left { fc + 1 } else { fc.saturating_sub(1) };
        let (golem, skel, spirit) =
            (my_side(toward_center, fr + 2), my_side(fc, fr + 1), my_side(fc, fr + 1));
        // Big pushes get the real defenders; medium ones the cheapest card that does the job.
        let options: Vec<(&str, (u32, u32))> = if air {
            vec![("musketeer", behind_tower(lane)), ("ice_spirit", spirit)]
        } else if value >= 4.0 {
            // Hog 2.6 guide: Musketeer is the core of every real defense.
            vec![("musketeer", behind_tower(lane)), ("cannon", center_pull(lane)), ("ice_golem", golem), ("skeletons", skel)]
        } else {
            vec![("ice_golem", golem), ("skeletons", skel), ("ice_spirit", spirit), ("cannon", center_pull(lane))]
        };
        options
            .iter()
            .filter(|(card, _)| cost(card).is_some_and(|c| c <= left))
            .find_map(|&(card, tile)| Self::deploy(s, card, tile, false, Why::Defend(lane)))
    }
}

impl Policy for HogCycle {
    fn decide(&mut self, s: &GameState) -> Option<Action> {
        if !s.in_battle || s.battle_time < Duration::from_secs(2) {
            return None;
        }
        // Right behind a fresh Hog Rider: pre-Log their log-able Hog defenders if one is in
        // their hand, else Ice Spirit.
        if let Some((t, lane)) = self.last_hog
            && s.battle_time.saturating_sub(t) < FOLLOW_UP
        {
            let d = enemy_deck(s);
            let loggable = LOG_KILLS.iter().any(|c| d.in_hand(c) == enemy::InHand::Likely);
            let (col, _) = bridge(lane);
            if loggable && let Some(a) = Self::deploy(s, "the_log", (col, 9), true, Why::FollowUp) {
                return Some(a);
            }
            if let Some(a) = Self::deploy(s, "ice_spirit", bridge(lane), false, Why::FollowUp) {
                return Some(a);
            }
        }

        if let Some(lane) = s.main_threat() {
            // Under attack: defend, and otherwise save elixir for the next defense.
            return self.defend(s, lane).or_else(|| {
                (s.elixir >= 10).then(|| self.defend(s, lane.other())).flatten()
            });
        }

        // The opponent just spent their elixir (recognized plays): punish in the other lane.
        if s.enemy_elixir.is_some_and(|e| e <= PUNISH_AT) && s.elixir >= COUNTER_HOG_AT {
            let busy = s.enemy_plays.last().filter(|p| s.battle_time.saturating_sub(p.t) < RECENT_PLAY).map(|p| Lane::of_col(p.tile.0));
            let lane = busy.map_or_else(|| self.push_lane(s), Lane::other);
            if let Some(a) = Self::deploy(s, "hog_rider", bridge(lane), false, Why::Push(lane)) {
                return Some(a);
            }
        }
        // Beatdown guide: a tank or pump dropped in their back is slow; Hog the other lane now.
        if s.elixir >= COUNTER_HOG_AT
            && let Some(p) = recent_play(s, |c| c.has_role("tank") || c.has_role("pump")).filter(|p| p.tile.1 <= ENEMY_BACK_ROW)
        {
            let lane = Lane::of_col(p.tile.0).other();
            if let Some(a) = Self::deploy(s, "hog_rider", bridge(lane), false, Why::Push(lane)) {
                return Some(a);
            }
        }
        // Siege: Hog Rider targets buildings; send it at their X-Bow or Mortar.
        if s.elixir >= COUNTER_HOG_AT
            && let Some(p) = recent_play(s, |c| c.has_role("building_siege"))
        {
            let lane = Lane::of_col(p.tile.0);
            if let Some(a) = Self::deploy(s, "hog_rider", bridge(lane), false, Why::Push(lane)) {
                return Some(a);
            }
        }
        // A tower 2-3 Fireballs from falling: finish it with spells (no risky Hog needed).
        if let Some(t) = s.towers
            && let Some((side, _)) = t.enemy.iter().enumerate().filter_map(|(i, h)| h.map(|h| (i, h))).filter(|(_, h)| *h <= FIREBALL_FINISH).min_by(|a, b| a.1.total_cmp(&b.1))
        {
            let lane = if side == 0 { Lane::Left } else { Lane::Right };
            if let Some(a) = Self::deploy(s, "fireball", enemy_tower(lane), true, Why::TowerSpell) {
                return Some(a);
            }
        }
        if s.battle_time < self.opening_wait && s.elixir < 10 {
            return None;
        }
        // Counter-push right after a defense (the opponent just spent elixir there), else
        // a normal push; double elixir pushes earlier.
        let counter = self.last_defense_lane.filter(|(t, _)| s.battle_time.saturating_sub(*t) < COUNTER_WINDOW);
        let hog_at = match counter {
            Some(_) => COUNTER_HOG_AT,
            None if s.battle_time >= DOUBLE_ELIXIR => self.hog_at.saturating_sub(1),
            None => self.hog_at,
        };
        // Ahead on towers: play safe, push only with spare elixir.
        let hog_at = if tower_lead(s) > 0 { hog_at + 1 } else { hog_at };
        // Hog 2.6 guide: out-cycle their Hog counters (Tesla, Inferno, Tornado ...).
        let counter_ready = blocking_hog_counters(s) && s.elixir < 10;
        if s.elixir >= hog_at && !counter_ready {
            let lane = counter.map_or_else(|| self.push_lane(s), |(_, l)| l);
            if let Some(a) = Self::deploy(s, "hog_rider", bridge(lane), false, Why::Push(lane)) {
                return Some(a);
            }
        }
        if s.elixir >= self.leak_at {
            let lane = self.push_lane(s);
            // Cycle cheap cards first to get Hog Rider back sooner; Musketeer behind the push
            // lane supports the next push; Cannon only as a last resort. Fireball stays in hand
            // for defense unless nothing else can stop the leak at full elixir.
            let options: [(&str, (u32, u32)); 5] = [
                ("ice_spirit", (9, 24)),
                ("skeletons", (8, 24)),
                ("ice_golem", (9, 24)),
                ("musketeer", behind_tower(lane)),
                ("cannon", center_pull(lane)),
            ];
            return options
                .iter()
                .find_map(|&(card, tile)| Self::deploy(s, card, tile, false, Why::Leak))
                .or_else(|| (s.elixir >= 10).then(|| Self::deploy(s, "fireball", enemy_tower(lane), true, Why::TowerSpell)).flatten());
        }
        None
    }

    fn on_action(&mut self, action: &Action, s: &GameState) {
        let Action::Deploy { card, col, why, .. } = action;
        match *why {
            Why::Push(lane) => self.last_hog = Some((s.battle_time, lane)),
            Why::FollowUp => self.last_hog = None,
            Why::Defend(lane) => {
                self.last_defense[lane_idx(lane)] = Some(s.battle_time);
                self.last_defense_lane = Some((s.battle_time, lane));
            }
            _ => {}
        }
        // Track defensive spending per lane (troops and spells on my side).
        if matches!(why, Why::Defend(_) | Why::Swarm) {
            let lane = Lane::of_col(*col);
            self.spent[lane_idx(lane)].push((s.battle_time, cost(card).unwrap_or(0)));
            self.spent[lane_idx(lane)].retain(|(t, _)| s.battle_time.saturating_sub(*t) < BUDGET_WINDOW);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(elixir: u8, hand: [&str; 4], secs: u64, enemies: &[(u32, u32)]) -> GameState {
        GameState {
            in_battle: true,
            elixir,
            elixir_fill: elixir as f32,
            hand: hand.map(|c| (!c.is_empty()).then(|| c.to_string())),
            next: None,
            battle_time: Duration::from_secs(secs),
            enemies: enemies.to_vec(),
            enemy_kinds: Vec::new(),
            ..Default::default()
        }
    }

    fn act(a: Option<Action>) -> Option<(String, u32, u32, Why)> {
        a.map(|Action::Deploy { card, col, row, why, .. }| (card, col, row, why))
    }

    fn enemy_play(card: &str, tile: (u32, u32), secs: u64) -> state::EnemyPlay {
        state::EnemyPlay { card: card.into(), tile, t: Duration::from_secs(secs), confidence: 0.9 }
    }

    #[test]
    fn punishes_a_spent_opponent_with_hog_in_the_other_lane() {
        let mut p = HogCycle::default();
        // They just dropped a big push on the left (still on their side): elixir ~1.
        let mut s = state(4, ["hog_rider", "cannon", "the_log", "fireball"], 40, &[(4, 6)]);
        s.enemy_plays = vec![enemy_play("pekka", (4, 6), 39)];
        s.enemy_elixir = Some(1.0);
        assert_eq!(act(p.decide(&s)), Some(("hog_rider".to_string(), 14, 17, Why::Push(Lane::Right))));
    }

    #[test]
    fn no_punish_when_they_still_have_elixir() {
        let mut p = HogCycle::default();
        let mut s = state(4, ["hog_rider", "cannon", "the_log", "fireball"], 40, &[]);
        s.enemy_elixir = Some(7.0);
        assert_eq!(act(p.decide(&s)), None, "4 elixir is below the normal hog threshold");
    }

    #[test]
    fn air_threat_gets_an_air_defender_not_cannon() {
        let mut p = HogCycle::default();
        let mut s = state(6, ["hog_rider", "cannon", "ice_golem", "musketeer"], 50, &[(3, 20)]);
        s.enemy_plays = vec![enemy_play("balloon", (3, 13), 47)];
        let (card, ..) = act(p.decide(&s)).unwrap();
        assert_eq!(card, "musketeer", "Cannon and Ice Golem cannot hit a Balloon");
    }

    #[test]
    fn air_swarm_is_not_logged() {
        let mut p = HogCycle::default();
        let swarm = [(13, 20), (14, 20), (14, 21), (15, 21)];
        let mut s = state(5, ["the_log", "ice_spirit", "hog_rider", "cannon"], 40, &swarm);
        s.enemy_plays = vec![enemy_play("minion_horde", (14, 14), 38)];
        let a = act(p.decide(&s));
        assert!(a.as_ref().is_none_or(|(c, ..)| c != "the_log" && c != "cannon"), "The Log and Cannon cannot hit air: {a:?}");
    }

    fn with_plays(mut s: GameState, plays: &[(&str, (u32, u32), u64)]) -> GameState {
        s.enemy_plays = plays.iter().map(|&(c, t, secs)| enemy_play(c, t, secs)).collect();
        s
    }

    #[test]
    fn guide_out_cycle_hog_counters() {
        // Their Tesla is back in hand (4 plays since): no plain Hog push into it.
        let mut p = HogCycle::default();
        let plays = [("tesla", (8, 10), 20), ("knight", (3, 12), 24), ("archers", (3, 4), 28), ("zap", (14, 20), 31), ("fireball", (14, 22), 34)];
        let s = with_plays(state(7, ["hog_rider", "cannon", "the_log", "skeletons"], 40, &[]), &plays);
        assert_eq!(act(p.decide(&s)), None, "Tesla in hand");
        // Tesla just played: out of cycle, Hog goes.
        let plays = [("knight", (3, 12), 24), ("archers", (3, 4), 28), ("zap", (14, 20), 31), ("tesla", (8, 10), 38)];
        let s = with_plays(state(7, ["hog_rider", "cannon", "the_log", "skeletons"], 40, &[]), &plays);
        assert_eq!(act(p.decide(&s)).map(|a| a.0), Some("hog_rider".into()));
    }

    #[test]
    fn guide_musketeer_first_on_big_pushes() {
        let mut p = HogCycle::default();
        let push = [(3, 19), (4, 19), (3, 20)];
        let s = state(7, ["cannon", "musketeer", "ice_golem", "hog_rider"], 50, &push);
        let (card, ..) = act(p.decide(&s)).unwrap();
        assert_eq!(card, "musketeer");
    }

    #[test]
    fn guide_punish_tank_in_the_back_other_lane() {
        let mut p = HogCycle::default();
        let mut s = with_plays(state(5, ["hog_rider", "cannon", "the_log", "fireball"], 40, &[(3, 3)]), &[("golem", (3, 3), 39)]);
        s.enemy_elixir = Some(4.0);
        assert_eq!(act(p.decide(&s)), Some(("hog_rider".to_string(), 14, 17, Why::Push(Lane::Right))));
    }

    #[test]
    fn guide_hog_the_siege_building() {
        let mut p = HogCycle::default();
        let mut s = with_plays(state(5, ["hog_rider", "cannon", "the_log", "fireball"], 40, &[(5, 12)]), &[("x_bow", (5, 12), 39)]);
        s.enemy_elixir = Some(4.0);
        assert_eq!(act(p.decide(&s)), Some(("hog_rider".to_string(), 3, 17, Why::Push(Lane::Left))));
    }

    #[test]
    fn guide_fireball_a_low_tower() {
        let mut p = HogCycle::default();
        let mut s = state(5, ["fireball", "cannon", "the_log", "skeletons"], 60, &[]);
        s.towers = Some(vision::towers::TowerHp { enemy: [Some(0.9), Some(0.08)], mine: [Some(1.0), Some(1.0)] });
        assert_eq!(act(p.decide(&s)), Some(("fireball".to_string(), 14, 6, Why::TowerSpell)));
    }

    #[test]
    fn guide_save_the_log_for_goblin_barrel() {
        let mut p = HogCycle::default();
        let plays = [("goblin_barrel", (3, 5), 20), ("knight", (3, 12), 24), ("archers", (3, 4), 28), ("zap", (14, 20), 31), ("princess", (14, 4), 34)];
        let swarm = [(13, 20), (14, 20), (14, 21)];
        let s = with_plays(state(5, ["the_log", "cannon", "hog_rider", "fireball"], 40, &swarm), &plays);
        let a = act(p.decide(&s));
        assert!(a.as_ref().is_none_or(|(c, ..)| c != "the_log"), "barrel in their hand: keep the Log: {a:?}");
    }

    #[test]
    fn guide_pre_log_with_hog() {
        let mut p = HogCycle::default();
        let plays = [("skeleton_army", (3, 12), 20), ("knight", (3, 12), 24), ("archers", (3, 4), 28), ("zap", (14, 20), 31), ("musketeer", (14, 4), 34)];
        let s = with_plays(state(7, ["hog_rider", "the_log", "ice_spirit", "cannon"], 40, &[]), &plays);
        let a = p.decide(&s).unwrap();
        let Action::Deploy { ref card, col: hog_col, .. } = a;
        assert_eq!(card, "hog_rider", "Skeleton Army dies to the Log: Hog goes despite it");
        p.on_action(&a, &s);
        let mut s2 = s.clone();
        s2.battle_time = Duration::from_millis(40_500);
        s2.elixir = 3;
        s2.hand = ["skeletons", "the_log", "ice_spirit", "cannon"].map(|c| Some(c.to_string()));
        let (card, col, row, why) = act(p.decide(&s2)).unwrap();
        assert_eq!((card.as_str(), why), ("the_log", Why::FollowUp), "Skeleton Army in their hand: pre-Log");
        assert!(row < 15 && col == hog_col, "in front of their tower, Hog's lane: ({col},{row})");
    }

    #[test]
    fn guide_play_safe_when_ahead() {
        let mut p = HogCycle::default();
        let mut s = state(6, ["hog_rider", "cannon", "the_log", "skeletons"], 60, &[]);
        assert_eq!(act(p.decide(&s)).map(|a| a.0), Some("hog_rider".into()));
        let mut p = HogCycle::default();
        s.towers = Some(vision::towers::TowerHp { enemy: [None, Some(0.8)], mine: [Some(1.0), Some(0.9)] });
        assert_eq!(act(p.decide(&s)), None, "a tower up: wait for more elixir");
    }

    #[test]
    fn pushes_the_weaker_enemy_tower() {
        let mut p = HogCycle::default();
        let mut s = state(7, ["hog_rider", "cannon", "the_log", "skeletons"], 60, &[]);
        s.towers = Some(vision::towers::TowerHp { enemy: [Some(0.4), Some(0.9)], mine: [Some(1.0), Some(1.0)] });
        assert_eq!(act(p.decide(&s)), Some(("hog_rider".to_string(), 3, 17, Why::Push(Lane::Left))));
        // Nearly even towers: keep the default lane.
        s.towers = Some(vision::towers::TowerHp { enemy: [Some(0.85), Some(0.9)], mine: [Some(1.0), Some(1.0)] });
        let mut p = HogCycle::default();
        assert_eq!(act(p.decide(&s)).map(|a| a.3), Some(Why::Push(Lane::Right)));
    }

    #[test]
    fn full_elixir_cycles_instead_of_wasting_fireball() {
        let mut p = HogCycle::default();
        let s = state(10, ["fireball", "ice_spirit", "the_log", "cannon"], 60, &[]);
        assert_eq!(act(p.decide(&s)).map(|a| a.0).as_deref(), Some("ice_spirit"), "Fireball stays for defense");
        // Only spells left: Fireball the tower rather than leak.
        let s = state(10, ["fireball", "the_log", "", ""], 60, &[]);
        assert_eq!(act(p.decide(&s)), Some(("fireball".to_string(), 14, 6, Why::TowerSpell)));
    }

    #[test]
    fn defends_the_threatened_lane_with_cannon_in_center() {
        let mut p = HogCycle::default();
        let s = state(6, ["hog_rider", "cannon", "the_log", "fireball"], 30, &[(3, 20)]);
        assert_eq!(act(p.decide(&s)), Some(("cannon".to_string(), 8, 21, Why::Defend(Lane::Left))));
    }

    #[test]
    fn no_hog_push_while_under_attack() {
        let mut p = HogCycle::default();
        let s = state(9, ["hog_rider", "the_log", "fireball", "ice_spirit"], 30, &[(14, 19)]);
        let a = act(p.decide(&s));
        assert!(a.as_ref().is_none_or(|(c, ..)| c != "hog_rider"), "{a:?}");
    }

    #[test]
    fn logs_a_swarm_from_behind() {
        let mut p = HogCycle::default();
        let swarm = [(13, 20), (14, 20), (14, 21), (15, 21)];
        let s = state(5, ["the_log", "cannon", "hog_rider", "musketeer"], 40, &swarm);
        let (card, _, row, why) = act(p.decide(&s)).unwrap();
        assert_eq!((card.as_str(), why), ("the_log", Why::Swarm));
        assert!(row >= 22, "log lands behind the group so it rolls through: row {row}");
    }

    #[test]
    fn enemies_on_their_own_side_are_not_threats() {
        let mut p = HogCycle::default();
        let s = state(7, ["hog_rider", "cannon", "", ""], 30, &[(14, 8), (3, 10)]);
        assert_eq!(act(p.decide(&s)).map(|a| a.0).as_deref(), Some("hog_rider"));
    }

    #[test]
    fn defense_cooldown_then_counter_push_same_lane() {
        let mut p = HogCycle::default();
        let s = state(6, ["cannon", "ice_golem", "hog_rider", "the_log"], 30, &[(3, 20)]);
        let a = p.decide(&s).unwrap();
        p.on_action(&a, &s);
        // Same single attacker 1 s later: cooldown, save elixir.
        let s = state(6, ["skeletons", "ice_golem", "hog_rider", "the_log"], 31, &[(4, 22)]);
        assert_eq!(p.decide(&s), None);
        // Threat gone: counter-push into the defended (left) lane.
        let s = state(7, ["skeletons", "ice_golem", "hog_rider", "the_log"], 34, &[]);
        assert_eq!(act(p.decide(&s)), Some(("hog_rider".to_string(), 3, 17, Why::Push(Lane::Left))));
    }

    #[test]
    fn hog_then_ice_spirit_follow_up_same_lane() {
        let mut p = HogCycle::default();
        let s = state(7, ["ice_spirit", "hog_rider", "cannon", "the_log"], 20, &[]);
        let a = p.decide(&s).unwrap();
        p.on_action(&a, &s);
        let s2 = state(3, ["ice_spirit", "", "cannon", "the_log"], 21, &[]);
        assert_eq!(act(p.decide(&s2)), Some(("ice_spirit".to_string(), 14, 17, Why::FollowUp)));
    }

    #[test]
    fn waits_in_opening_unless_full() {
        let mut p = HogCycle::default();
        assert_eq!(p.decide(&state(8, ["hog_rider", "cannon", "", ""], 3, &[])), None);
        assert_eq!(act(p.decide(&state(10, ["hog_rider", "cannon", "", ""], 3, &[]))).map(|a| a.0).as_deref(), Some("hog_rider"));
    }

    #[test]
    fn troops_respect_cooldown_even_against_a_swarm() {
        let mut p = HogCycle::default();
        let swarm = [(13, 20), (14, 20), (14, 21), (15, 21)];
        // No spell in hand: first troop goes in...
        let s = state(9, ["cannon", "ice_golem", "skeletons", "musketeer"], 40, &swarm);
        let a = p.decide(&s).unwrap();
        p.on_action(&a, &s);
        // ...but not a second one half a second later.
        let s = state(6, ["ice_spirit", "ice_golem", "skeletons", "musketeer"], 40, &swarm);
        assert_eq!(p.decide(&s), None);
    }

    #[test]
    fn defense_budget_caps_spending() {
        assert_eq!(defense_budget(3.5), 4);
        assert_eq!(defense_budget(9.0), 7);
        let mut p = HogCycle::default();
        // One unknown attacker (~3.5): budget 4, Cannon (3) fits, Musketeer (4) only if nothing spent.
        // Cannon spent at 34.5 s; at 38 s the 3 s cooldown is over but the 4 s budget window isn't.
        let mut earlier = state(9, ["cannon", "", "", ""], 34, &[]);
        earlier.battle_time += Duration::from_millis(500);
        p.on_action(
            &Action::Deploy { slot: 0, card: "cannon".into(), col: 9, row: 21, enemy_half: false, why: Why::Defend(Lane::Right) },
            &earlier,
        );
        let s = state(9, ["musketeer", "", "", ""], 38, &[(14, 20)]);
        assert_eq!(p.decide(&s), None, "3 of 4 already spent in the window");
        let s = state(9, ["skeletons", "", "", ""], 38, &[(14, 20)]);
        assert_eq!(act(p.decide(&s)).map(|a| a.0).as_deref(), Some("skeletons"), "1 elixir still fits");
    }

    #[test]
    fn counter_push_at_low_elixir_after_defense() {
        let mut p = HogCycle::default();
        let s = state(5, ["cannon", "hog_rider", "", ""], 30, &[(3, 20)]);
        let a = p.decide(&s).unwrap();
        p.on_action(&a, &s);
        let s = state(4, ["skeletons", "hog_rider", "", ""], 35, &[]);
        assert_eq!(act(p.decide(&s)), Some(("hog_rider".to_string(), 3, 17, Why::Push(Lane::Left))));
    }

    #[test]
    fn leak_cycles_cheap_cards_first() {
        let mut p = HogCycle::default();
        let s = state(9, ["cannon", "musketeer", "ice_spirit", "the_log"], 40, &[]);
        assert_eq!(act(p.decide(&s)).map(|a| a.0).as_deref(), Some("ice_spirit"));
    }

    #[test]
    fn two_known_swarm_units_get_logged() {
        let mut p = HogCycle::default();
        let mut s = state(5, ["the_log", "cannon", "hog_rider", "musketeer"], 40, &[(14, 20), (15, 21)]);
        // Two Barbarians (one 5-elixir card) together: Log them.
        s.enemy_kinds = vec![Some("barbarians".into()), Some("barbarians".into())];
        assert_eq!(act(p.decide(&s)).map(|a| (a.0, a.3)), Some(("the_log".to_string(), Why::Swarm)));
        // Two Goblins (2 elixir): the towers handle it, no card.
        s.enemy_kinds = vec![Some("goblins".into()), Some("goblins".into())];
        assert_eq!(p.decide(&s), None);
        // Same two units, unknown type: a troop defends instead (Musketeer first, Hog 2.6 guide).
        s.enemy_kinds = vec![None, None];
        assert_eq!(act(p.decide(&s)).map(|a| a.0).as_deref(), Some("musketeer"));
    }

    #[test]
    fn towers_handle_cheap_pushes() {
        let mut p = HogCycle::default();
        let mut s = state(6, ["cannon", "skeletons", "ice_golem", "the_log"], 40, &[(14, 20), (14, 21), (15, 20)]);
        s.enemy_kinds = vec![Some("skeletons".into()); 3];
        assert_eq!(p.decide(&s), None, "3 skeletons (1 elixir) die to the tower");
    }

    #[test]
    fn push_values() {
        let sk = |n: usize| (0..n).map(|i| ((10 + i as u32 % 3, 20 + i as u32 / 3), Some("skeletons"))).collect::<Vec<_>>();
        assert_eq!(push_value(&sk(3)), 1.0);
        assert_eq!(push_value(&sk(12)), 3.0, "skeleton army");
        assert_eq!(push_value(&[((14, 20), Some("hog_rider"))]), 4.0);
        assert_eq!(push_value(&[((14, 20), None)]), 3.5);
        assert_eq!(push_value(&[((14, 20), None), ((14, 21), None), ((3, 25), None)]), 7.5, "a pair + a single");
    }

    #[test]
    fn medium_push_gets_a_cheap_answer() {
        let mut p = HogCycle::default();
        let mut s = state(8, ["cannon", "musketeer", "ice_golem", "skeletons"], 40, &[(14, 20)]);
        s.enemy_kinds = vec![Some("elixir_golem".into())];
        assert_eq!(act(p.decide(&s)).map(|a| a.0).as_deref(), Some("ice_golem"));
    }

    #[test]
    fn card_table_from_assets() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/cards.json");
        let t = Cards::from_json(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(t.len(), 122 + 4, "122 cards + 4 tower troops");
        let e = |s: &str| t.get(s).unwrap_or_else(|| panic!("{s} missing")).elixir;
        for (card, c) in [("skeletons", 1), ("ice_spirit", 1), ("the_log", 2), ("ice_golem", 2), ("knight", 3),
            ("cannon", 3), ("hog_rider", 4), ("musketeer", 4), ("fireball", 4), ("mini_pekka", 4), ("giant", 5),
            ("pekka", 7), ("golem", 8), ("three_musketeers", 9)]
        {
            assert_eq!(e(card), Some(c), "{card}");
        }
        assert_eq!(e("mirror"), None);
        for s in ["skeletons", "goblins", "barbarians", "skeleton_army", "minion_horde"] {
            assert!(t.get(s).unwrap().swarm, "{s} is a swarm");
        }
        for s in ["hog_rider", "knight", "cannon", "fireball"] {
            assert!(!t.get(s).unwrap().swarm, "{s} is not a swarm");
        }
        // Every playable card has a cost (Mirror copies one; tower troops aren't played).
        let troops = t.iter().filter(|c| c.kind.as_deref() == Some("Tower Troop")).count();
        assert_eq!(troops, 4);
        assert!(t.iter().all(|c| c.elixir.is_some_and(|e| (1..=10).contains(&e))
            || c.slug == "mirror"
            || c.kind.as_deref() == Some("Tower Troop")));
        // The embedded table is that file.
        assert_eq!(cards().len(), t.len());
    }

    #[test]
    fn densest_group() {
        assert_eq!(densest(&[(1, 1), (2, 1), (1, 2), (10, 10)], 2.5).map(|g| g.2), Some(3));
        assert_eq!(densest(&[], 2.5), None);
    }
}
