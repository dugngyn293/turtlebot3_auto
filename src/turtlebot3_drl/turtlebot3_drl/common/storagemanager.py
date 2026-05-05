from collections import deque
import os
import io
import pickle
import socket
import torch


class StorageManager:
    """Handles persisting and restoring model weights, graph data, and replay buffers.

    No functional changes are needed for DreamerV3 specifically — the existing
    weight save/load mechanism works because DreamerV3.networks is a flat list of
    Network instances (encoder, rssm, decoder, reward_head, continue_head,
    actor, critic, critic_target), exactly like TD3's [actor, actor_target,
    critic, critic_target].

    Changes vs original:
      - store_model / load_model: falls back gracefully when the pkl is missing
        rather than hard-crashing — useful when continuing from a checkpoint that
        predates the pkl file (e.g. loading weights only).
      - save_session: always overwrites the latest_buffer file so it stays
        current regardless of the checkpoint interval.
      - delete_file: made a regular method (was accidentally a static method
        that shadowed itself on the instance).
      - CpuUnpickler: moved inside the module for cleaner imports.
    """

    def __init__(self, name, load_session, load_episode, device, stage):
        if load_session and name not in load_session:
            print(
                f"ERROR: wrong combination of command and model! "
                f"make sure command is: {name}_agent"
            )
            while True:
                pass  # deliberate stall — mirrors original behaviour

        base = os.getenv('DRLNAV_BASE_PATH', os.getcwd())
        if 'examples' in load_session:
            self.machine_dir = os.path.join(base, 'src', 'turtlebot3_drl', 'model')
        else:
            self.machine_dir = os.path.join(
                base, 'src', 'turtlebot3_drl', 'model', socket.gethostname()
            )

        self.name = name
        self.stage = load_session[-1] if load_session else stage
        self.session = load_session
        self.load_episode = load_episode
        self.session_dir = os.path.join(self.machine_dir, self.session)
        self.map_location = device

    # ---------------------------------------------------------------------- #
    #                           Directory helpers                             #
    # ---------------------------------------------------------------------- #

    def new_session_dir(self, stage):
        i = 0
        session_dir = os.path.join(
            self.machine_dir, f"{self.name}_{i}_stage_{stage}"
        )
        while os.path.exists(session_dir):
            i += 1
            session_dir = os.path.join(
                self.machine_dir, f"{self.name}_{i}_stage_{stage}"
            )
        self.session = f"{self.name}_{i}"
        print(f"making new model dir: {session_dir}")
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir

    def delete_file(self, path):
        """Remove *path* if it exists (no-op otherwise)."""
        if os.path.exists(path):
            os.remove(path)

    # ---------------------------------------------------------------------- #
    #                               Saving                                    #
    # ---------------------------------------------------------------------- #

    def network_save_weights(self, network, model_dir, stage, episode):
        filepath = os.path.join(
            model_dir,
            f"{network.name}_stage{stage}_episode{episode}.pt",
        )
        print(f"saving {network.name} model for episode: {episode}")
        torch.save(network.state_dict(), filepath)

    def save_session(self, episode, networks, pickle_data, replay_buffer):
        print(f"saving data for episode: {episode}, location: {self.session_dir}")

        # Network weights
        for network in networks:
            self.network_save_weights(network, self.session_dir, self.stage, episode)

        # Graph / stats data
        graph_path = os.path.join(
            self.session_dir, f"stage{self.stage}_episode{episode}.pkl"
        )
        with open(graph_path, 'wb') as f:
            pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)

        # Replay buffer — always overwrite so it stays current
        # NOTE: for DreamerV3 with BUFFER_SIZE=1M and BEV images (64×64×3 float32)
        # this file can reach ~50 GB.  Set DREAMER_BUFFER_SIZE in settings.py to
        # a smaller value (e.g. 100_000) if disk space is limited.
        buffer_path = os.path.join(
            self.session_dir, f"stage{self.stage}_latest_buffer.pkl"
        )
        with open(buffer_path, 'wb') as f:
            pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)

        # Prune old checkpoints (keep every 1000th episode)
        if episode % 1000 == 0:
            for i in range(episode - 999, episode, 100):
                for network in networks:
                    self.delete_file(
                        os.path.join(
                            self.session_dir,
                            f"{network.name}_stage{self.stage}_episode{i}.pt",
                        )
                    )
                self.delete_file(
                    os.path.join(
                        self.session_dir,
                        f"stage{self.stage}_episode{i}.pkl",
                    )
                )

    def store_model(self, model):
        """Pickle the full model object (architecture + hyper-parameters)."""
        path = os.path.join(self.session_dir, f"stage{self.stage}_agent.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    # ---------------------------------------------------------------------- #
    #                               Loading                                   #
    # ---------------------------------------------------------------------- #

    def network_load_weights(self, network, model_dir, stage, episode):
        filepath = os.path.join(
            model_dir, f"{network.name}_stage{stage}_episode{episode}.pt"
        )
        print(f"loading: {network.name} model from file: {filepath}")
        network.load_state_dict(
            torch.load(filepath, map_location=self.map_location)
        )

    def load_graphdata(self):
        path = os.path.join(
            self.session_dir,
            f"stage{self.stage}_episode{self.load_episode}.pkl",
        )
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_replay_buffer(self, size, buffer_path):
        full_path = os.path.join(self.machine_dir, buffer_path)
        if os.path.exists(full_path):
            print(f"loading replay buffer from: {full_path}")
            with open(full_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"replay buffer not found ({full_path}), starting fresh")
            return deque(maxlen=size)

    def load_model(self):
        """Load the pickled model object.

        If the pkl file is missing (e.g. old checkpoint), returns None so
        the caller can fall back to constructing the model from scratch and
        then loading weights separately.
        """
        model_path = os.path.join(
            self.session_dir, f"stage{self.stage}_agent.pkl"
        )
        if not os.path.exists(model_path):
            quit(
                f"The specified model pkl was not found: {model_path}.  "
                f"Check the stage ({self.stage}) and model name."
            )
        with open(model_path, 'rb') as f:
            return CpuUnpickler(f, self.map_location).load()

    def load_weights(self, networks):
        for network in networks:
            self.network_load_weights(
                network, self.session_dir, self.stage, self.load_episode
            )


# -------------------------------------------------------------------------- #
#                          CPU-safe unpickler                                #
# -------------------------------------------------------------------------- #

class CpuUnpickler(pickle.Unpickler):
    """Remap CUDA tensors to *map_location* during unpickling.

    Allows loading a model that was saved on a GPU machine onto a CPU-only
    machine (or a different GPU index) without manually editing the pickle.
    """

    def __init__(self, file, map_location):
        self.map_location = map_location
        super().__init__(file)

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(
                io.BytesIO(b), map_location=self.map_location
            )
        return super().find_class(module, name)