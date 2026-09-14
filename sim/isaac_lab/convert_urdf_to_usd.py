"""Convert NexArm URDF to NVIDIA Omniverse USD for Isaac Sim / Isaac Lab.

Usage with Isaac Sim python runner:
    ./python.sh sim/isaac_lab/convert_urdf_to_usd.py --input sim/fusion_export/nexarm.urdf --output sim/isaac_lab/nexarm.usd
"""

import argparse
import os
import sys


def convert_urdf_to_usd(input_urdf: str, output_usd: str) -> None:
    try:
        import omni.kit.commands
        from omni.isaac.urdf import _urdf
    except ImportError:
        print(
            "ERROR: This script must be run with the Isaac Sim Python environment "
            "(e.g., './python.sh convert_urdf_to_usd.py' inside Isaac Sim directory)."
        )
        sys.exit(1)

    input_urdf = os.path.abspath(input_urdf)
    output_usd = os.path.abspath(output_usd)

    if not os.path.exists(input_urdf):
        raise FileNotFoundError(f"Input URDF not found: {input_urdf}")

    os.makedirs(os.path.dirname(output_usd), exist_ok=True)

    # Set URDF import parameters
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = True
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.import_inertia_tensor = True
    import_config.distance_scale = 1.0
    import_config.density = 0.0
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

    print(f"Importing URDF: {input_urdf} -> {output_usd}...")
    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=input_urdf,
        import_config=import_config,
        dest_path=output_usd,
    )
    print(f"Successfully exported USD to {output_usd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert URDF to USD for Isaac Sim / Isaac Lab")
    parser.add_argument("--input", default="sim/fusion_export/nexarm.urdf", help="Path to input .urdf")
    parser.add_argument("--output", default="sim/isaac_lab/nexarm.usd", help="Path to output .usd")
    args = parser.parse_args()

    convert_urdf_to_usd(args.input, args.output)
