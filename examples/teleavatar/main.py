#!/usr/bin/env python3
"""
Main entry point for running Teleavatar robot with OpenPI policy.

This script uses the standard openpi_client.runtime framework for clean,
modular robot control with remote policy inference.

Usage:
    # Start policy server first (in another terminal):
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_teleavatar \
        --policy.dir=checkpoints/pi05_teleavatar/my_experiment/20000

    # Then run this script:
    python examples/teleavatar/main.py --remote-host 127.0.0.1
"""

"""
3种 control mode 切换 (通过 --control-mode 参数选择)

  # 方式 1: Pure Chunk (ActionChunkBroker)
  python main.py --control-mode action_chunk

  # 方式 2: Receding Horizon (RecedingHorizonBroker)
  python main.py --control-mode receding_horizon

  # 方式 3: Soft Inpainting (SoftInpaintingBroker)
  python main.py --control-mode soft_inpainting \
      --actions-to-execute 26 --actions-to-keep 4

  注意：Soft Inpainting 需要服务端配置:
  - use_soft_inpainting=True
  - 可选: use_correlated_noise=True (需要预计算相关性矩阵)
"""

import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import receding_horizon_broker
from openpi_client import soft_inpainting_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro
import sys
from pathlib import Path
# Add project root to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.teleavatar import env as _env


@dataclasses.dataclass
class Args:
    """Command-line arguments for Teleavatar deployment."""

    # Remote policy server settings
    remote_host: str = "0.0.0.0"
    """IP address of the policy server (e.g., '192.168.1.100')"""

    remote_port: int = 8000
    """Port of the policy server"""

    # Control settings
    control_frequency: float = 20.0
    """Control loop frequency in Hz (default: 15 Hz, matching DROID)"""

    action_horizon: int = 30
    """Number of actions in each chunk returned by policy (default: 10)"""

    open_loop_horizon: int = 24
    """Number of actions to execute before querying policy again (default: 8)"""

    # Control mode selection
    control_mode: str = "action_chunk"
    """Control mode: 'action_chunk', 'receding_horizon', or 'soft_inpainting'"""

    # Soft Inpainting settings (only used when control_mode='soft_inpainting')
    actions_to_execute: int = 13
    """For soft_inpainting: number of actions to execute before re-querying"""

    actions_to_keep: int = 2
    """For soft_inpainting: number of actions to keep for inpainting constraint"""

    # Task settings
    prompt: str = "Stack the three blocks'"
    """Language instruction for the robot"""

    # Episode settings
    num_episodes: int = 100
    """Number of episodes to run"""

    max_episode_steps: int = 250
    """Maximum steps per episode (0 = unlimited)"""


def main(args: Args) -> None:
    """Main function to run Teleavatar with policy inference."""

    logging.info("=" * 60)
    logging.info("Teleavatar OpenPI Deployment")
    logging.info("=" * 60)
    logging.info(f"Policy server: ws://{args.remote_host}:{args.remote_port}")
    logging.info(f"Control frequency: {args.control_frequency} Hz")
    logging.info(f"Control mode: {args.control_mode}")
    logging.info(f"Action horizon: {args.action_horizon} steps")
    logging.info(f"Open-loop horizon: {args.open_loop_horizon} steps")
    if args.control_mode == "soft_inpainting":
        logging.info(f"Actions to execute: {args.actions_to_execute} steps")
        logging.info(f"Actions to keep: {args.actions_to_keep} steps")
    logging.info(f"Prompt: '{args.prompt}'")
    logging.info("=" * 60)

    # Validate settings
    if args.open_loop_horizon > args.action_horizon:
        logging.warning(
            f"open_loop_horizon ({args.open_loop_horizon}) > action_horizon ({args.action_horizon}). "
            f"This means the policy will be queried before the previous chunk is exhausted."
        )

    # Create WebSocket client policy
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.remote_host,
        port=args.remote_port,
    )

    # Get and log server metadata
    metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Connected to policy server. Metadata: {metadata}")

    # Create Teleavatar environment
    # Note: Images are kept at original resolution to match training data
    environment = _env.TeleavatarEnvironment(
        prompt=args.prompt,
    )

    # Create policy agent based on control mode
    if args.control_mode == "action_chunk":
        # Pure Action Chunk: execute all action_horizon steps before re-querying
        broker = action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.open_loop_horizon,
        )
        logging.info("Using ActionChunkBroker (Pure Chunk mode)")
    elif args.control_mode == "receding_horizon":
        # Receding Horizon: execute open_loop_horizon steps, discard the rest
        broker = receding_horizon_broker.RecedingHorizonBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
            open_loop_horizon=args.open_loop_horizon,
        )
        logging.info("Using RecedingHorizonBroker (Receding Horizon mode)")
    elif args.control_mode == "soft_inpainting":
        # Soft Inpainting: execute actions_to_execute steps, keep actions_to_keep for inpainting
        broker = soft_inpainting_broker.SoftInpaintingBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
            actions_to_execute=args.actions_to_execute,
            actions_to_keep=args.actions_to_keep,
        )
        logging.info("Using SoftInpaintingBroker (Soft Inpainting mode)")
    else:
        raise ValueError(
            f"Unknown control_mode: {args.control_mode}. "
            f"Must be 'action_chunk', 'receding_horizon', or 'soft_inpainting'."
        )

    agent = _policy_agent.PolicyAgent(policy=broker)

    # Create runtime
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],  # Could add video recording, logging, etc.
        max_hz=args.control_frequency,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    # Run!
    logging.info("\nStarting robot control loop...")
    logging.info("Press Ctrl+C to stop\n")

    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("\n\nStopping robot (Ctrl+C pressed)...")
    finally:
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )

    # Parse arguments and run
    args: Args = tyro.cli(Args)
    main(args)
