# Trossen bimanual rope rig — how to use it

Operating summary for the dual-arm Trossen WidowX AI setup used for rope
(deformable linear object) manipulation.

**Read section 2 before writing any control code.** Most of those rules were
learned by breaking the rig, and several look arbitrary until you know the
failure they prevent.

Documentation only — no model weights, no evaluation code. The scripts
referenced here live in the rig repository (`missaltim`, under
`rope_rig/scripts/`), alongside the full 740-line `rope_rig/HANDOFF.md` this
summarises.

---

## 1. The rig, in facts

| Thing | Value |
|---|---|
| Arms | Trossen WidowX AI, ROS 2 Humble, driver + firmware **1.8.4** |
| Leaders | `192.168.1.2`, `192.168.1.3` |
| Followers | `192.168.1.4`, `192.168.1.5` |
| Cameras | 4x RealSense D405 @ 640x480, aligned depth |
| Wrist cams | `.4` arm -> `218622270778`; `.5` arm -> `218622274511` |

**Naming trap.** `rig.yaml` calls the `.2`/`.4` pair "left"; the operator calls
that their **right** arm. The scripts use the operator's convention
(`ARMS = {'right': '192.168.1.4', 'left': '192.168.1.5'}`, mapping in
`rig_tf.py`). Get this wrong and everything mirrors.

**Geometry and calibration** (verified on hardware — do not re-derive):

- Bases are **0.92 m apart** on side rails, facing each other (yaw -+90 deg).
- Bases sit **5 mm above** the tabletop (`rig.yaml base_mount z: 0.005`),
  verified with a tape measure. An earlier wrist-camera depth fit claiming
  -0.04 was wrong by 4 cm.
- World z = 0 is the tabletop. A cord lying on it reads ~8 mm (its top surface).

---

## 2. How to drive the arms

1. **Never use `InterpolationSpace.cartesian`.** It is *sticky driver state*.
   Once set, the daemon runs a full IK solve on every comms cycle forever; near
   a tool-down pose that IK is ill-conditioned, a UDP read fails within 1-2
   minutes, the daemon dies, the controller idles and **the arm sags**. Use
   Cartesian *goals* with `InterpolationSpace.joint`.
2. **`num_trajectory_check_samples=0`.** The 1.8.4 default of 1000 means 1000
   IK solves per call while holding the driver's data mutex.
3. **One command per waypoint, observe at ~5 Hz.** Do not stream Cartesian
   setpoints at 200 Hz — each call solves IK in your thread under the mutex and
   starves the daemon. Teleop streams flat out only because it sends *joint*
   positions, which need no IK.
4. **Never `blocking=True`.** There is no timeout; a non-converging move blocks
   forever and no signal can interrupt it — only SIGKILL, which then wedges the
   controller. Use flag-only signal handlers.
5. **Never `pkill -9` an arm client.** That is what leaves ghost sessions. Note
   `pkill -f` / `pgrep -f` also match the invoking shell itself; the safe form
   is `kill -INT $(ps -eo pid,cmd | grep "[p]attern" | awk '{print $1}')`.
6. **Verify mode changes, never assume them.** Use
   `arm_stream.ensure_modes(d, Mode.position)`, which sets and reads back with
   `get_modes()` until the controller confirms. Prefer one `set_all_modes` over
   separate arm and gripper calls, which race.
7. **Never command the gripper fully closed.** It saturates motor 6 at -70 N
   and the controller then *silently ignores arm motion*. Grasp with
   `set_gripper_external_effort(-12 N, goal_time=4.0)` — a slow squeeze; a fast
   one bounces the object out. Open to ~17 mm to grasp, ~30 mm at overview
   poses (closed fingers occlude the rope).
8. **`connect()` needs a watchdog.** `configure()` can block forever with no
   exception. Before concluding a controller is wedged, check for your own
   stray processes — one client per controller, and a second `configure()`
   hangs rather than failing.

**Never run camera code in the arm-control process.** A wrist D405 stall
recovered in-process corrupted the heap and core-dumped, killing the driver
clients with an arm deployed. Grabs go through `grab_once.py` in a subprocess.

---

## 3. Safety

- The arms face each other 0.92 m apart with each one's +x reach pointing at
  the other. **Both at base x=0.38 simultaneously leaves 1 cm of clearance.**
  Gate every simultaneous motion with `collision.py` (`swept_clearance`).
- The required envelope gap was **raised from 0.06 m to 0.105 m** by the
  operator after watching a probe walk in. Do not lower it.
- **Sequential operation is inherently safe** — one arm parked while the other
  works. That is what the grasp loop does.
- **Nothing can prevent an arm sagging when comms die.** The controller idles
  and torque drops by design, and that is exactly when the host has lost
  control. Mitigate by working low with something soft under the workspace —
  not in software.
- Arms held at the overview pose are held only *while that process lives*.
  Ctrl-C parks them; if the process is gone they are limp. Check before
  assuming.

---

## 4. Known failure modes

**Daemon death.** Driver 1.8.4 stores the daemon exception and never restarts
the thread, so every later call rethrows forever. Three independent causes were
confirmed by disassembly:

| Cause | Status | Notes |
|---|---|---|
| Cartesian IK under lock | fixed | avoided by the joint-space pattern above |
| Gripper mode churn | present | races at position <-> external_effort transitions |
| Link-side bursts | present | hit during pure idle, after a loss burst |

**The USB NIC renegotiating to half duplex** caused five deaths in one
afternoon, including during pure joint-space holds with nothing in flight.
Check it before any arm work:

```bash
ethtool <arm-nic>            # must read Duplex: Full
sudo ethtool -r <arm-nic>    # renegotiate; park the arms first
```

**Recovery:** a fresh `configure()` recovered every death observed — no latched
error, no power cycle needed. Keep only one live client at a time.

**Log-reading traps.** ~25-30% "messages lost" is *normal* here: the driver's
UDP read has a 1 ms timeout and counts each timeout as lost, so "no warning" is
not "no loss". `Failed to read UDP message due to Success` is a driver
formatting bug, not a network diagnosis — do not tune the network. Never filter
the driver's log lines; doing so hid every loss warning and produced repeated
wrong "zero packet loss" reports.

**Upgrade.** 1.8.4 has a logging deadlock (fixed 1.8.6); 1.9.1 makes getters
return copies rather than references into daemon memory; **1.9.3 + matching
firmware** makes the IK pre-check opt-in and fixes the post-kill controller
wedge. Given the death rate this is urgent rather than optional.

---

## 5. Core scripts

| Script | Purpose |
|---|---|
| `arm_stream.py` | control primitives: `connect`, `ensure_modes`, `stream_to`, `park`, logging |
| `collision.py` | capsule model of both arms; `clearance()`, `swept_clearance()` |
| `rig_tf.py` | camera -> ee -> base -> world chain; `ray_to_plane()` |
| `go_overview.py` | put both arms at the overview pose and hold |
| `arm_daemon.py` | persistent two-arm executor; holds overview, runs moves with no parks between cycles, self-healing |
| `grab_once.py` | single RGBD grab in an isolated subprocess |
| `pnp2.py` | verified pick-and-place |
| `grasp_test.py` | verified perception-driven pick, lift, nudge, drop, park |

**Verified working:** pick-and-place of a ~3.7 mm cable in 31 s with zero comms
deaths; task-space accuracy ~0.1-0.3 mm; reaches the tabletop with no kinematic
floor. Earlier "silent no-op" moves were the saturated gripper (rule 7) and IK
failures that stayed invisible until driver logging was attached.

---

## 6. Working rules

- **Never write code in `/tmp`.** Power-cycling controllers is a routine
  recovery step here, so reboots are frequent; a full day of work was nearly
  lost this way.
- Report measurements honestly, including when a previous conclusion was wrong.
  Several confident diagnoses on this rig were wrong and had to be retracted —
  say so plainly when it happens.

---

## 7. Model deployment notes

`docs/` holds the per-model notes for running learned controllers on this rig.

| Topic | Note |
|---|---|
| Running the closed-loop controller | [`docs/HANDOFF_ROBOT_AGENT.md`](docs/HANDOFF_ROBOT_AGENT.md) |
| Pre-session readiness | [`docs/REAL_ROBOT_READINESS.md`](docs/REAL_ROBOT_READINESS.md) |
| The planner | [`docs/HANDOFF_CD_PLANNER.md`](docs/HANDOFF_CD_PLANNER.md) |
| Ours — base and small GRU | [`docs/HANDOFF_OURS_BASE_MODEL.md`](docs/HANDOFF_OURS_BASE_MODEL.md), [`docs/HANDOFF_GRU_SMALL_RIG.md`](docs/HANDOFF_GRU_SMALL_RIG.md) |
| GA-Net | [`docs/HANDOFF_GANET_MPC.md`](docs/HANDOFF_GANET_MPC.md) |
| IN-BiLSTM and LSTM-GCN | [`docs/HANDOFF_INBILSTM_LSTMGCN_RIG.md`](docs/HANDOFF_INBILSTM_LSTMGCN_RIG.md) |
| Knot untangling | [`docs/HANDOFF_KNOT_UNTANGLE_RIG.md`](docs/HANDOFF_KNOT_UNTANGLE_RIG.md) |
