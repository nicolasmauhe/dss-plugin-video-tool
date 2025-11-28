
import dataiku
import tempfile
import os
import cv2
import base64
from dataiku.llm.agent_tools import BaseAgentTool


class VideoAnalysisTool(BaseAgentTool):
    """
    A tool that enables an Agent to 'watch' a video and answer specific questions about it.
    It handles frame extraction and vision processing internally.
    """

    def set_config(self, config, plugin_config):
        self.config = config
        self.folder = dataiku.Folder(self.config.get("input_folder", "videos"))
        self.client = dataiku.api_client()
        self.llm = self.client.get_default_project().get_llm(self.config["llm_id"])

    def get_descriptor(self, tool):
        """
        Defines the inputs the Main Agent must provide.
        """
        return {
            "description": "Use this tool to analyze the visual content of a video file. You must provide the filename and a specific question. If the filename is incorrect the tool will return the list of available filenames.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "video_name": {
                        "type": "string",
                        "description": "The exact name of the video file (e.g., 'my_video.mp4')"
                    },
                    "question": {
                        "type": "string",
                        "description": "The specific question you want to answer about the video (e.g., 'How many people are in the video?')"
                    }
                },
                "required": ["video_name", "question"]
            }
        }

    def invoke(self, input, trace):
        """
        Main Logic: Validation -> Extraction -> Vision LLM Call
        """
        # 1. Parse Arguments
        args = input.get("input", {})
        target_name = args.get("video_name", "").strip()
        question = args.get("question", "Describe this video.")

        # 2. Validate Video Existence
        # We remove leading slashes just in case
        all_files = [p.lstrip('/') for p in self.folder.list_paths_in_partition()]
        
        # Check for exact match or return list of available videos
        if target_name in all_files:
            matched_file = target_name
        else:
            file_list_str = ", ".join(all_files)
            return {
                "output": f"Error: Video '{target_name}' was not found. Available videos are: [{file_list_str}]. Please retry with a valid name.",
                "sources": []
            }

        # 3. Process Video (Extract Frames)
        try:
            frames_b64 = self._extract_frames(matched_file, max_frames=6)
        except Exception as e:
            return {"output": f"Technical Error processing video file: {str(e)}", "sources": []}

        # 4. Call Internal Vision LLM
        try:
            completion = self.llm.new_completion()
            
            # Construct the prompt for the Vision Model
            completion.with_message("You are an expert video analyst.", "system")
            
            mp_message = completion.new_multipart_message('user')
            mp_message.with_text(f"Analyze these frames to answer this question: {question}")
            
            for b64 in frames_b64:
                mp_message.with_inline_image(b64)
            
            mp_message.add()
            
            # Execute
            response = completion.execute()
            
            # Return the description to the Main Agent
            return {
                "output": f"Visual Analysis of '{matched_file}': {response.text}",
                "sources": []
            }

        except Exception as e:
            return {"output": f"Error calling Vision LLM: {str(e)}", "sources": []}

    # --- HELPER METHODS ---

    def _extract_frames(self, filename, max_frames=10):
        """Downloads video to temp, extracts N frames, converts to base64"""
        frames_b64 = []
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            stream = self.folder.get_download_stream(filename)
            tmp_file.write(stream.read())
            tmp_path = tmp_file.name
        
        try:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames)
            
            count = 0
            processed = 0
            while cap.isOpened() and processed < max_frames:
                ret, frame = cap.read()
                if not ret: break
                
                if count % step == 0:
                    # Optional: Resize to reduce tokens/latency (Standardize to 512px width)
                    height, width = frame.shape[:2]
                    if width > 512:
                        new_h = int(height * (512 / width))
                        frame = cv2.resize(frame, (512, new_h))

                    _, buffer = cv2.imencode('.jpg', frame)
                    b64_str = base64.b64encode(buffer).decode('utf-8')
                    frames_b64.append(b64_str)
                    processed += 1
                count += 1
            cap.release()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        return frames_b64

