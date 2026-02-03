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
        
        # Graph layout
        self.graph_margin = 40
        self.graph_top = 100  # Space for text info above graph
        
    def update_track_depth(self, track_id: int, frame_idx: int, depth_meters: Optional[float]):
        """Update the depth history for a track (depth should be in meters)."""
        if track_id not in self.depth_history:
            self.depth_history[track_id] = deque(maxlen=self.history_length)
        
        if depth_meters is not None and depth_meters > 0:
            self.depth_history[track_id].append((frame_idx, depth_meters))
    
    def prune_old_tracks(self, current_frame: int):
        """Remove tracks that haven't been seen recently."""
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
    
    def draw(
        self,
        frame_height: int,
        frame_idx: int,
        num_tracks: int,
        latency_ms: float,
    ) -> np.ndarray:
        """
        Draw the info panel.
        
        Args:
            frame_height: Height of the main frame (panel will match)
            frame_idx: Current frame index
            num_tracks: Current number of active tracks
            latency_ms: Processing latency in milliseconds
            
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
        y_text += line_height + 10
        
        # Draw depth graph
        self._draw_depth_graph(panel, frame_idx, y_text)
        
        return panel
    
    def _draw_depth_graph(self, panel: np.ndarray, current_frame: int, y_start: int):
        """Draw the depth over time graph."""
        h, w = panel.shape[:2]
        margin = self.graph_margin
        
        # Graph area
        graph_left = margin
        graph_right = w - margin // 2
        graph_top = y_start + 20
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


def concatenate_with_panel(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    """Concatenate the main frame with the info panel on the right."""
    return np.hstack([frame, panel])
