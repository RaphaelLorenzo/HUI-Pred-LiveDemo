"""Visualization utilities for cv2 drawing."""
import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a consistent color for a track ID using golden ratio."""
    hue = (track_id * 0.618033988749895) % 1.0
    rgb = cv2.cvtColor(np.array([[[int(hue * 180), 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


class InfoPanel:
    """Side panel for displaying tracking information and depth graphs."""
    
    def __init__(
        self,
        width: int = 300,
        history_length: int = 100,
        min_track_appearances: int = 32,
        y_max_meters: float = 6.0,
    ):
        """
        Initialize the info panel.
        
        Args:
            width: Panel width in pixels
            history_length: Number of timesteps to keep in history
            min_track_appearances: Minimum appearances in history to show in graph
            y_max_meters: Maximum depth in meters for the y-axis
        """
        self.width = width
        self.history_length = history_length
        self.min_track_appearances = min_track_appearances
        self.y_max_meters = y_max_meters
        
        # Track depth history: {track_id: deque of (frame_idx, depth_meters)}
        self.depth_history: Dict[int, deque] = {}
        
        # Track IP output history: {track_id: deque of (frame_idx, ip_value)}
        self.ip_history: Dict[int, deque] = {}
        
        # Graph layout
        self.graph_margin = 40
        self.graph_top = 100  # Space for text info above graph
        
    def update_track_depth(self, track_id: int, frame_idx: int, depth_meters: Optional[float]):
        """Update the depth history for a track (depth should be in meters)."""
        if track_id not in self.depth_history:
            self.depth_history[track_id] = deque(maxlen=self.history_length)
        
        if depth_meters is not None and depth_meters > 0:
            self.depth_history[track_id].append((frame_idx, depth_meters))
    
    def update_track_ip(self, track_id: int, frame_idx: int, ip_value):
        """Update the IP output history for a track. Invalid values are mapped to 0."""
        if track_id not in self.ip_history:
            self.ip_history[track_id] = deque(maxlen=self.history_length)
        
        # Map invalid values (non-float) to 0
        if isinstance(ip_value, (int, float)):
            val = float(ip_value)
        else:
            val = 0.0
        
        self.ip_history[track_id].append((frame_idx, val))
    
    def prune_old_tracks(self, current_frame: int):
        """Remove tracks that haven't been seen recently."""
        # Prune depth history
        tracks_to_remove = []
        for track_id, history in self.depth_history.items():
            if len(history) == 0:
                tracks_to_remove.append(track_id)
                continue
            # Check if the most recent entry is too old
            last_frame = history[-1][0]
            if current_frame - last_frame > self.history_length:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.depth_history[track_id]
        
        # Prune IP history
        tracks_to_remove = []
        for track_id, history in self.ip_history.items():
            if len(history) == 0:
                tracks_to_remove.append(track_id)
                continue
            last_frame = history[-1][0]
            if current_frame - last_frame > self.history_length:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.ip_history[track_id]
    
    def draw(
        self,
        frame_height: int,
        frame_idx: int,
        num_tracks: int,
        latency_ms: float,
        ip_latency_ms: float = 0.0,
    ) -> np.ndarray:
        """
        Draw the info panel.
        
        Args:
            frame_height: Height of the main frame (panel will match)
            frame_idx: Current frame index
            num_tracks: Current number of active tracks
            latency_ms: Processing latency in milliseconds
            ip_latency_ms: Interaction prediction latency in milliseconds
            
        Returns:
            Panel image as numpy array (BGR)
        """
        # Create panel with dark background
        panel = np.zeros((frame_height, self.width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Dark gray background
        
        # Draw text info at top
        y_text = 25
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_color = (220, 220, 220)
        
        cv2.putText(panel, f"Frame: {frame_idx}", (10, y_text), font, font_scale, text_color, 1)
        y_text += line_height
        cv2.putText(panel, f"Tracks: {num_tracks}", (10, y_text), font, font_scale, text_color, 1)
        y_text += line_height
        cv2.putText(panel, f"Latency: {latency_ms:.1f} ms", (10, y_text), font, font_scale, text_color, 1)
        y_text += line_height
        cv2.putText(panel, f"Int.Pred. Latency: {ip_latency_ms:.1f} ms", (10, y_text), font, font_scale, text_color, 1)
        y_text += line_height + 10
        
        # Calculate space for two graphs
        available_height = frame_height - y_text - self.graph_margin
        graph_height_each = (available_height - 30) // 2  # 30px gap between graphs
        
        # Draw depth graph (top)
        self._draw_depth_graph(panel, frame_idx, y_text, graph_height_each)
        
        # Draw IP output graph (bottom)
        ip_graph_start = y_text + graph_height_each + 50  # 50px gap for title and spacing
        self._draw_ip_graph(panel, frame_idx, ip_graph_start, graph_height_each)
        
        return panel
    
    def _draw_depth_graph(self, panel: np.ndarray, current_frame: int, y_start: int, max_height: int = None):
        """Draw the depth over time graph."""
        h, w = panel.shape[:2]
        margin = self.graph_margin
        
        # Graph area
        graph_left = margin
        graph_right = w - margin // 2
        graph_top = y_start + 20
        if max_height is not None:
            graph_bottom = y_start + max_height
        else:
            graph_bottom = h - margin
        graph_width = graph_right - graph_left
        graph_height = graph_bottom - graph_top
        
        if graph_height < 50 or graph_width < 50:
            return  # Not enough space
        
        # Draw title
        cv2.putText(panel, "Depth (m) vs Time", (margin, y_start + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Draw graph background
        cv2.rectangle(panel, (graph_left, graph_top), (graph_right, graph_bottom), (50, 50, 50), -1)
        cv2.rectangle(panel, (graph_left, graph_top), (graph_right, graph_bottom), (100, 100, 100), 1)
        
        # X-axis: sliding window of last history_length frames
        x_min = max(0, current_frame - self.history_length + 1)
        x_max = current_frame
        
        # Y-axis: 0 to y_max_meters
        y_min = 0
        y_max = self.y_max_meters
        
        # Draw grid lines and labels
        self._draw_grid(panel, graph_left, graph_right, graph_top, graph_bottom,
                        x_min, x_max, y_min, y_max)
        
        # Find tracks with enough appearances
        eligible_tracks = []
        for track_id, history in self.depth_history.items():
            # Count appearances in the current window
            appearances = sum(1 for f, d in history if x_min <= f <= x_max)
            if appearances >= self.min_track_appearances:
                eligible_tracks.append(track_id)
        
        # Plot each eligible track
        for track_id in eligible_tracks:
            color = get_track_color(track_id)
            history = self.depth_history[track_id]
            
            for frame, depth in history:
                if frame < x_min or frame > x_max:
                    continue
                if depth < y_min or depth > y_max:
                    continue
                
                # Convert to pixel coordinates
                px = int(graph_left + (frame - x_min) / max(1, x_max - x_min) * graph_width)
                py = int(graph_bottom - (depth - y_min) / (y_max - y_min) * graph_height)
                
                # Draw point
                cv2.circle(panel, (px, py), 3, color, -1)
    
    def _draw_ip_graph(self, panel: np.ndarray, current_frame: int, y_start: int, max_height: int = None):
        """Draw the IP output over time graph (0 to 1 scale)."""
        h, w = panel.shape[:2]
        margin = self.graph_margin
        
        # Graph area
        graph_left = margin
        graph_right = w - margin // 2
        graph_top = y_start + 20
        if max_height is not None:
            graph_bottom = y_start + max_height
        else:
            graph_bottom = h - margin
        graph_width = graph_right - graph_left
        graph_height = graph_bottom - graph_top
        
        if graph_height < 50 or graph_width < 50:
            return  # Not enough space
        
        # Draw title
        cv2.putText(panel, "IP Output vs Time", (margin, y_start + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Draw graph background
        cv2.rectangle(panel, (graph_left, graph_top), (graph_right, graph_bottom), (50, 50, 50), -1)
        cv2.rectangle(panel, (graph_left, graph_top), (graph_right, graph_bottom), (100, 100, 100), 1)
        
        # X-axis: sliding window of last history_length frames
        x_min = max(0, current_frame - self.history_length + 1)
        x_max = current_frame
        
        # Y-axis: 0 to 1 for IP output
        y_min = 0.0
        y_max = 1.0
        
        # Draw grid lines and labels (custom for 0-1 range)
        self._draw_ip_grid(panel, graph_left, graph_right, graph_top, graph_bottom,
                           x_min, x_max, y_min, y_max)
        
        # Find tracks with enough appearances
        eligible_tracks = []
        for track_id, history in self.ip_history.items():
            # Count appearances in the current window
            appearances = sum(1 for f, v in history if x_min <= f <= x_max)
            if appearances >= self.min_track_appearances:
                eligible_tracks.append(track_id)
        
        # Plot each eligible track
        for track_id in eligible_tracks:
            color = get_track_color(track_id)
            history = self.ip_history[track_id]
            
            for frame, ip_val in history:
                if frame < x_min or frame > x_max:
                    continue
                # Clamp ip_val to [0, 1]
                ip_val = max(0.0, min(1.0, ip_val))
                
                # Convert to pixel coordinates
                px = int(graph_left + (frame - x_min) / max(1, x_max - x_min) * graph_width)
                py = int(graph_bottom - (ip_val - y_min) / (y_max - y_min) * graph_height)
                
                # Draw point
                cv2.circle(panel, (px, py), 3, color, -1)
    
    def _draw_ip_grid(
        self,
        panel: np.ndarray,
        left: int, right: int, top: int, bottom: int,
        x_min: int, x_max: int, y_min: float, y_max: float
    ):
        """Draw grid lines and axis labels for IP graph (0-1 range)."""
        grid_color = (70, 70, 70)
        label_color = (150, 150, 150)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        
        # Y-axis grid lines (IP output 0-1)
        y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        for y_val in y_ticks:
            py = int(bottom - (y_val - y_min) / (y_max - y_min) * (bottom - top))
            cv2.line(panel, (left, py), (right, py), grid_color, 1)
            cv2.putText(panel, f"{y_val:.2f}", (2, py + 4), font, font_scale, label_color, 1)
        
        # X-axis: show a few frame markers
        width = right - left
        num_x_ticks = 5
        for i in range(num_x_ticks + 1):
            x_val = x_min + (x_max - x_min) * i / num_x_ticks
            px = int(left + width * i / num_x_ticks)
            cv2.line(panel, (px, top), (px, bottom), grid_color, 1)
            if i % 2 == 0:  # Label every other tick
                cv2.putText(panel, f"{int(x_val)}", (px - 10, bottom + 12), font, font_scale, label_color, 1)
    
    def _draw_grid(
        self,
        panel: np.ndarray,
        left: int, right: int, top: int, bottom: int,
        x_min: int, x_max: int, y_min: float, y_max: float
    ):
        """Draw grid lines and axis labels."""
        grid_color = (70, 70, 70)
        label_color = (150, 150, 150)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        
        # Y-axis grid lines (depth)
        y_ticks = [0, 1, 2, 3, 4, 5, 6]
        for y_val in y_ticks:
            if y_val < y_min or y_val > y_max:
                continue
            py = int(bottom - (y_val - y_min) / (y_max - y_min) * (bottom - top))
            cv2.line(panel, (left, py), (right, py), grid_color, 1)
            cv2.putText(panel, f"{y_val:.0f}", (5, py + 4), font, font_scale, label_color, 1)
        
        # X-axis: show a few frame markers
        width = right - left
        num_x_ticks = 5
        for i in range(num_x_ticks + 1):
            x_val = x_min + (x_max - x_min) * i / num_x_ticks
            px = int(left + width * i / num_x_ticks)
            cv2.line(panel, (px, top), (px, bottom), grid_color, 1)
            if i % 2 == 0:  # Label every other tick
                cv2.putText(panel, f"{int(x_val)}", (px - 10, bottom + 12), font, font_scale, label_color, 1)


def concatenate_with_panel(frame: np.ndarray, panel: np.ndarray, *extra_panels: np.ndarray) -> np.ndarray:
    """Concatenate the main frame with the info panel(s) on the right."""
    if extra_panels:
        return np.hstack([frame, panel, *extra_panels])
    return np.hstack([frame, panel])


class BehaviorPanel:
    """
    Side panel for behavior prototypes (e.g. lights Idle -> Offering interaction).
    First third: two round lights (blue -> green) with engagement.
    Second third: two eyes — almost closed/bored when idle, wide open and expressive when engaged.
    Third: reserved for future behavior.
    """

    # Idle = low-intensity blue (BGR)
    LIGHT_IDLE_COLOR = (180, 100, 40)   # dim blue
    # Offering = bright green (BGR)
    LIGHT_OFFERING_COLOR = (80, 255, 80)  # bright green

    def __init__(self, width: int = 220):
        self.width = width

    def draw(
        self,
        frame_height: int,
        lights_scale: float,
    ) -> np.ndarray:
        """
        Draw the behavior panel.

        Args:
            frame_height: Height of the main frame (panel will match).
            lights_scale: 0.0 = idle (blue), 1.0 = offering (green). Linear blend
                          of color and intensity between idle and offering.

        Returns:
            Panel image as numpy array (BGR).
        """
        lights_scale = max(0.0, min(1.0, float(lights_scale)))
        panel = np.zeros((frame_height, self.width, 3), dtype=np.uint8)
        panel[:] = (28, 28, 28)

        # Panel divided in three horizontal bands (for 3 behavior types)
        third = frame_height // 3
        # First third: "Lights" behavior — two round lights
        self._draw_lights_section(panel, 0, third, lights_scale)
        # Second third: "Eyes" behavior — two eyes (bored/closed -> wide open)
        self._draw_eyes_section(panel, third, 2 * third, lights_scale)
        # Third: placeholder for future behavior
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "Behavior 3", (10, 2 * third + 30), font, 0.45, (120, 120, 120), 1)

        # Title at top
        cv2.putText(panel, "Behaviors", (10, 22), font, 0.6, (220, 220, 220), 1)
        return panel

    def _draw_lights_section(self, panel: np.ndarray, y_start: int, y_end: int, lights_scale: float):
        """Draw the first behavior: two round lights (blue -> green) in the given vertical band."""
        h_section = y_end - y_start
        if h_section < 40:
            return
        cx = self.width // 2
        cy = y_start + h_section // 2
        radius = min(28, (self.width - 40) // 4, h_section // 4)
        gap = max(20, radius)
        left_cx = cx - gap // 2 - radius
        right_cx = cx + gap // 2 + radius

        # Interpolate color: idle blue -> offering green
        b0, g0, r0 = self.LIGHT_IDLE_COLOR
        b1, g1, r1 = self.LIGHT_OFFERING_COLOR
        b = int(b0 + (b1 - b0) * lights_scale)
        g = int(g0 + (g1 - g0) * lights_scale)
        r = int(r0 + (r1 - r0) * lights_scale)
        # Intensity: start low (idle), end bright (offering)
        intensity = 0.35 + 0.65 * lights_scale
        color = (int(b * intensity), int(g * intensity), int(r * intensity))

        for px in (left_cx, right_cx):
            cv2.circle(panel, (px, cy), radius, color, -1)
            border_color = (min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40))
            cv2.circle(panel, (px, cy), radius, border_color, 1)

        cv2.putText(panel, "Lights", (10, y_start + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        label = "Idle" if lights_scale < 0.1 else ("Offering" if lights_scale > 0.9 else "Transition")
        cv2.putText(panel, label, (10, y_end - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1)

    def _draw_eyes_section(self, panel: np.ndarray, y_start: int, y_end: int, engagement_scale: float):
        """
        Draw the second behavior: two eyes that go from almost closed/bored (idle)
        to wide open and positively expressive (engaged).
        """
        h_section = y_end - y_start
        if h_section < 50:
            return
        cx = self.width // 2
        cy = y_start + h_section // 2
        eye_half_gap = 24
        left_cx = cx - eye_half_gap
        right_cx = cx + eye_half_gap

        ax_x = 14
        ax_y_idle = 2
        ax_y_engaged = 11
        ax_y = ax_y_idle + (ax_y_engaged - ax_y_idle) * engagement_scale
        angle_idle = -12
        angle_engaged = 3
        angle = angle_idle + (angle_engaged - angle_idle) * engagement_scale

        eye_outer_idle = (70, 70, 75)
        eye_outer_engaged = (245, 245, 250)
        b = int(eye_outer_idle[0] + (eye_outer_engaged[0] - eye_outer_idle[0]) * engagement_scale)
        g = int(eye_outer_idle[1] + (eye_outer_engaged[1] - eye_outer_idle[1]) * engagement_scale)
        r = int(eye_outer_idle[2] + (eye_outer_engaged[2] - eye_outer_idle[2]) * engagement_scale)
        eye_color = (b, g, r)

        for ex in (left_cx, right_cx):
            if engagement_scale < 0.15:
                pt1 = (ex - ax_x, cy)
                pt2 = (ex + ax_x, cy)
                cv2.line(panel, pt1, pt2, (90, 90, 95), 2)
            else:
                cv2.ellipse(
                    panel, (ex, cy), (int(ax_x), int(ax_y)), angle, 0, 360, eye_color, -1
                )
                cv2.ellipse(
                    panel, (ex, cy), (int(ax_x), int(ax_y)), angle, 0, 360, (60, 60, 65), 1
                )
                if engagement_scale > 0.5:
                    iris_radius = int(4 + 3 * engagement_scale)
                    iris_offset_y = int(2 * engagement_scale)
                    cv2.circle(panel, (ex, cy + iris_offset_y), iris_radius, (200, 160, 80), -1)
                    cv2.circle(panel, (ex, cy + iris_offset_y), iris_radius, (50, 50, 60), 1)
                    if engagement_scale > 0.7:
                        cv2.circle(panel, (ex + 2, cy + iris_offset_y - 2), 2, (255, 255, 255), -1)

        cv2.putText(panel, "Eyes", (10, y_start + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        label = "Bored" if engagement_scale < 0.15 else ("Engaged" if engagement_scale > 0.85 else "Opening")
        cv2.putText(panel, label, (10, y_end - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1)
