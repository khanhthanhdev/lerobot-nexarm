# Isaac Sim replay

Run `runner.py` with the Isaac Sim 5.1 interpreter declared in the video2sim
manifest. The runner loads the room, references the generated robot USD,
resolves every DOF by name, and replays canonical actions through the
articulation controller. LeRobot's base environment can import none of Isaac.
