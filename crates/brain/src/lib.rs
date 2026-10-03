//! Decision making: the `Policy` trait and a rule-based Hog 2.6 bot.

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
    /// Fireball on a tower at full elixir.
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

/// Elixir cost of known cards.
pub fn cost(card: &str) -> Option<u8> {
    Some(match card {
        "ice_spirit" | "skeletons" => 1,
        "the_log" | "ice_golem" => 2,
        "cannon" | "archers" | "arrows" | "minions" | "knight" => 3,
        "hog_rider" | "musketeer" | "fireball" | "mini_pekka" => 4,
        "giant" => 5,
        _ => return None,
    })
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
/// 2. push: Hog Rider (+ Ice Spirit) when my side is clear, into the lane just defended;
/// 3. fallbacks: Fireball the tower at full elixir, play a card before elixir overflows.
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

/// Elixir allowed on defense in one lane per window: bigger pushes get more, never everything.
fn defense_budget(attackers: usize) -> u8 {
    (3.0 + 1.5 * attackers as f32).min(7.0) as u8
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

    fn spent_recently(&self, s: &GameState, lane: Lane) -> u8 {
        self.spent[lane_idx(lane)]
            .iter()
            .filter(|(t, _)| s.battle_time.saturating_sub(*t) < BUDGET_WINDOW)
            .map(|(_, c)| c)
            .sum()
    }

    fn defend(&self, s: &GameState, lane: Lane) -> Option<Action> {
        let threats = s.threats(lane);
        // Most advanced attacker (closest to my towers).
        let &(fc, fr) = threats.iter().max_by_key(|t| t.1)?;

        // Swarm: 3+ units together, or 2+ known swarm units (classifier) together.
        let swarmy: Vec<(u32, u32)> = s
            .threat_kinds(lane)
            .into_iter()
            .filter(|(_, k)| k.is_some_and(|k| SWARM_KINDS.contains(&k)))
            .map(|(t, _)| t)
            .collect();
        let group = densest(&threats, SWARM_RADIUS)
            .filter(|g| g.2 >= SWARM_MIN)
            .or_else(|| densest(&swarmy, SWARM_RADIUS).filter(|g| g.2 >= 2));
        if let Some((gc, gr, _)) = group {
            // The Log rolls up the arena from where it lands: drop it just behind the group.
            if let Some(a) = Self::deploy(s, "the_log", my_side(gc, gr + 2), false, Why::Swarm) {
                return Some(a);
            }
            if let Some(a) = Self::deploy(s, "fireball", (gc, gr), gr < MY_FIRST_ROW_HINT, Why::Swarm) {
                return Some(a);
            }
        }

        // Troops: one per lane per cooldown, and within the lane's elixir budget.
        let recent = self.last_defense[lane_idx(lane)].is_some_and(|t| s.battle_time.saturating_sub(t) < DEFENSE_COOLDOWN);
        if recent {
            return None;
        }
        let left = defense_budget(threats.len()).saturating_sub(self.spent_recently(s, lane));
        // Toward the center from the attacker, so it gets pulled between both towers.
        let toward_center = if lane == Lane::Left { fc + 1 } else { fc.saturating_sub(1) };
        let options: [(&str, (u32, u32)); 5] = [
            ("cannon", center_pull(lane)),
            ("ice_golem", my_side(toward_center, fr + 2)),
            ("skeletons", my_side(fc, fr + 1)),
            ("musketeer", behind_tower(lane)),
            ("ice_spirit", my_side(fc, fr + 1)),
        ];
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
        // Ice Spirit right behind a fresh Hog Rider.
        if let Some((t, lane)) = self.last_hog
            && s.battle_time.saturating_sub(t) < FOLLOW_UP
            && let Some(a) = Self::deploy(s, "ice_spirit", bridge(lane), false, Why::FollowUp)
        {
            return Some(a);
        }

        if let Some(lane) = s.main_threat() {
            // Under attack: defend, and otherwise save elixir for the next defense.
            return self.defend(s, lane).or_else(|| {
                (s.elixir >= 10).then(|| self.defend(s, lane.other())).flatten()
            });
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
        if s.elixir >= hog_at {
            let lane = counter.map_or(self.default_lane, |(_, l)| l);
            if let Some(a) = Self::deploy(s, "hog_rider", bridge(lane), false, Why::Push(lane)) {
                return Some(a);
            }
        }
        if s.elixir >= 10
            && let Some(a) = Self::deploy(s, "fireball", enemy_tower(self.default_lane), true, Why::TowerSpell)
        {
            return Some(a);
        }
        if s.elixir >= self.leak_at {
            let lane = self.default_lane;
            // Cycle cheap cards first to get Hog Rider back sooner; Musketeer behind the push
            // lane supports the next push; Cannon only as a last resort.
            let options: [(&str, (u32, u32)); 5] = [
                ("ice_spirit", (9, 24)),
                ("skeletons", (8, 24)),
                ("ice_golem", (9, 24)),
                ("musketeer", behind_tower(lane)),
                ("cannon", center_pull(lane)),
            ];
            return options.iter().find_map(|&(card, tile)| Self::deploy(s, card, tile, false, Why::Leak));
        }
        None
    }

    fn on_action(&mut self, action: &Action, s: &GameState) {
        let Action::Deploy { why, .. } = action;
        match *why {
            Why::Push(lane) => self.last_hog = Some((s.battle_time, lane)),
            Why::FollowUp => self.last_hog = None,
            Why::Defend(lane) => {
                self.last_defense[lane_idx(lane)] = Some(s.battle_time);
                self.last_defense_lane = Some((s.battle_time, lane));
            }
            Why::Swarm => {}
            _ => {}
        }
        // Track defensive spending per lane (troops and spells on my side).
        let Action::Deploy { card, col, why, .. } = action;
        if matches!(why, Why::Defend(_) | Why::Swarm) {
            let lane = Lane::of_col(*col);
            self.spent[lane_idx(lane)].push((s.battle_time, cost(card).unwrap_or(0)));
            self.spent[lane_idx(lane)].retain(|(t, _)| s.battle_time.saturating_sub(*t) < BUDGET_WINDOW);
        }
        match *why {
            _ => {}
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
        }
    }

    fn act(a: Option<Action>) -> Option<(String, u32, u32, Why)> {
        a.map(|Action::Deploy { card, col, row, why, .. }| (card, col, row, why))
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
        assert_eq!(defense_budget(1), 4);
        assert_eq!(defense_budget(4), 7);
        let mut p = HogCycle::default();
        // One attacker: budget 4, Cannon (3) fits, Musketeer (4) only if nothing spent.
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
        s.enemy_kinds = vec![Some("goblins".into()), Some("goblins".into())];
        assert_eq!(act(p.decide(&s)).map(|a| (a.0, a.3)), Some(("the_log".to_string(), Why::Swarm)));
        // Same two units, unknown type: a troop defends instead.
        s.enemy_kinds = vec![None, None];
        assert_eq!(act(p.decide(&s)).map(|a| a.0).as_deref(), Some("cannon"));
    }

    #[test]
    fn densest_group() {
        assert_eq!(densest(&[(1, 1), (2, 1), (1, 2), (10, 10)], 2.5).map(|g| g.2), Some(3));
        assert_eq!(densest(&[], 2.5), None);
    }
}
