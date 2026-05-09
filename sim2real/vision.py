import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

TagData = dict[int, dict]  # {tag_id: {"pos": np.ndarray, "rpy": np.ndarray}}


class AprilTagPose:
    """
    Detects AprilTags and returns poses in world coordinates relative to a base tag.

    Filtering (per tag):
      1. Outlier rejection — if the new reading jumps more than `max_jump_m` metres
         from the last accepted pose, it is discarded and the last known pose is returned.
      2. EMA smoothing — accepted readings are blended with the running estimate:
             pos = alpha * new + (1 - alpha) * previous
         Lower alpha = smoother / more lag. Higher alpha = more responsive / more noise.
    """

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2
    LINE_HEIGHT = 18
    # AprilTag world -> simulation world:
    # sim_x = -tag_y, sim_y = tag_x, sim_z = -tag_z
    _SIM_FRAME_REMAP = np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        base_id: int = 12,
        tag_size: float = 0.022,
        cam_index: int = 2,
        smooth_alpha: float = 0.2,
        max_jump_m: float = 0.5,
        max_missed_frames: int = 5,
        base_pos_offset_m: tuple[float, float, float] = (0.0, -0.10, 0.0),
    ):
        self.base_id = base_id
        self.tag_size = tag_size
        self.smooth_alpha = smooth_alpha
        self.max_jump_m = max_jump_m
        self.max_missed_frames = max(1, int(max_missed_frames))
        self.base_pos_offset_m = np.asarray(base_pos_offset_m, dtype=np.float64)

        self._T_base_cam: np.ndarray | None = None
        self._smooth: dict[int, dict] = {}  # last accepted world-frame pose per tag
        self._missed_frames: dict[int, int] = {}

        self.camera_matrix = np.array(
            [[1601.5, 0.0, 725.3], [0.0, 2398.4, 384.8], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist_coeffs = np.array(
            [0.1216, 0.5667, -0.0112, 0.0252, -4.2311], dtype=np.float64
        )

        self.detector = Detector(families="tag36h11")
        self.cap = cv2.VideoCapture(cam_index)

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_transform(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t.flatten()
        return T

    @staticmethod
    def _invert_transform(T: np.ndarray) -> np.ndarray:
        T_inv = np.eye(4)
        T_inv[:3, :3] = T[:3, :3].T
        T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
        return T_inv

    @staticmethod
    def _rotation_to_euler(R_mat: np.ndarray) -> np.ndarray:
        return R.from_matrix(R_mat).as_euler("xyz", degrees=True)

    @classmethod
    def _remap_to_sim_frame(
        cls, pos: np.ndarray, rot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        remap = cls._SIM_FRAME_REMAP
        sim_pos = remap @ np.asarray(pos, dtype=np.float64).reshape(3)
        sim_rot = remap @ np.asarray(rot, dtype=np.float64) @ remap.T
        return sim_pos, sim_rot

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def _filter(self, tag_id: int, pos: np.ndarray, rpy: np.ndarray) -> dict:
        """
        Outlier rejection + EMA smoothing.
        Returns the smoothed estimate. If the reading is an outlier,
        returns the last known good pose unchanged.
        """
        missed_frames = self._missed_frames.get(tag_id, 0)
        if tag_id not in self._smooth or missed_frames >= self.max_missed_frames:
            # First sighting or a fresh re-detection after the tag was missing:
            # accept the new measurement directly instead of comparing with stale data.
            self._smooth[tag_id] = {"pos": pos.copy(), "rpy": rpy.copy()}
            self._missed_frames[tag_id] = 0
            return self._smooth[tag_id]

        prev = self._smooth[tag_id]
        jump = float(np.linalg.norm(pos - prev["pos"]))

        if jump > self.max_jump_m:
            # Outlier — discard, return last known good pose
            return prev

        # EMA blend
        a = self.smooth_alpha
        self._smooth[tag_id] = {
            "pos": a * pos + (1 - a) * prev["pos"],
            "rpy": a * rpy + (1 - a) * prev["rpy"],
        }
        self._missed_frames[tag_id] = 0
        return self._smooth[tag_id]

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def get_tag_poses(
        self, show_window: bool = False
    ) -> tuple[TagData, np.ndarray | None]:
        ret, frame = self.cap.read()
        if not ret:
            return {}, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_params = [
            self.camera_matrix[0, 0],
            self.camera_matrix[1, 1],
            self.camera_matrix[0, 2],
            self.camera_matrix[1, 2],
        ]

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=cam_params,
            tag_size=self.tag_size,
        )
        detected_ids = {det.tag_id for det in detections}

        for tid in list(self._smooth):
            if tid in detected_ids:
                self._missed_frames[tid] = 0
            else:
                self._missed_frames[tid] = self._missed_frames.get(tid, 0) + 1

        # Update world-from-camera transform whenever base tag is visible
        for det in detections:
            if det.tag_id == self.base_id:
                T_cam_base = self._make_transform(det.pose_R, det.pose_t)
                self._T_base_cam = self._invert_transform(T_cam_base)
                break

        # If base tag has never been seen, nothing to return yet
        if self._T_base_cam is None:
            if show_window:
                cv2.putText(
                    frame,
                    "Waiting for base tag...",
                    (10, 30),
                    self.FONT,
                    self.FONT_SCALE,
                    (0, 0, 255),
                    self.FONT_THICKNESS,
                )
                cv2.imshow("AprilTag Vision", frame)
                cv2.waitKey(1)
            return {}, frame

        tag_data: TagData = {}

        for det in detections:
            T_cam_tag = self._make_transform(det.pose_R, det.pose_t)
            T_world_tag = self._T_base_cam @ T_cam_tag
            tag_pos = T_world_tag[:3, 3]
            tag_rot = T_world_tag[:3, :3]
            sim_pos, sim_rot = self._remap_to_sim_frame(tag_pos, tag_rot)

            raw_pos = sim_pos + self.base_pos_offset_m
            raw_rpy = self._rotation_to_euler(sim_rot)

            # Apply outlier rejection + EMA — this is what stops the pose flips
            filtered = self._filter(det.tag_id, raw_pos, raw_rpy)

            x_m, y_m, z_m = filtered["pos"]
            roll, pitch, yaw = filtered["rpy"]

            tag_data[det.tag_id] = {
                "pos": filtered["pos"].copy(),
                "rpy": filtered["rpy"].copy(),
            }

            self._draw_tag_overlay(
                frame, det, "[WORLD]", x_m, y_m, z_m, roll, pitch, yaw
            )

        # For tags not detected this frame, return their last known filtered pose
        for tid, data in self._smooth.items():
            if tid not in tag_data:
                tag_data[tid] = data.copy()

        if show_window:
            cv2.imshow("AprilTag Vision", frame)
            cv2.waitKey(1)
        # print(f"Detected tags: {list(tag_data.keys())}")
        return tag_data, frame

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_tag_overlay(self, frame, det, label, x, y, z, roll, pitch, yaw) -> None:
        tx, ty = int(det.corners[0][0]), int(det.corners[0][1]) - 10
        lh = self.LINE_HEIGHT
        f, s, t = self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
        lines = [
            (f"{label} ID:{det.tag_id}", (255, 255, 255)),
            (f"X: {x:.3f}m", (0, 0, 255)),
            (f"Y: {y:.3f}m", (0, 255, 0)),
            (f"Z: {z:.3f}m", (255, 0, 0)),
            (f"R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f}", (0, 255, 255)),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (tx, ty + i * lh), f, s, color, t)

    def release(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------


def main():
    vision = AprilTagPose(
        base_id=6,
        cam_index=2,
        smooth_alpha=0.5,  # lower = smoother, higher = more responsive
        max_jump_m=0.15,  # max plausible jump between frames (metres)
    )
    try:
        while True:
            tags, _ = vision.get_tag_poses(show_window=True)
            for tid, data in tags.items():
                p = data["pos"]
                print(f"ID {tid} -> X:{p[0]:.3f} Y:{p[1]:.3f} Z:{p[2]:.3f}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vision.release()
        print("Vision test complete.")


if __name__ == "__main__":
    main()
