"""Visual scoring module for prototype comparison in Vision2Web"""

import os
import re
import json
import time
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI

from vision2web.evaluation.prompts import VISUAL_PROMPT


class VisualScorer:
    """Scores visual fidelity by comparing prototype vs actual screenshots.

    Uses a VLM judge model (via OpenAI-compatible API) to compare
    a prototype image against an actual page screenshot.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        visual_model: str,
        log_dir: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.visual_model = visual_model
        self.logger = logging.getLogger('VisualScorer')

    def _parse_prompt_with_images(
        self,
        prompt: str,
        images: Dict[str, str],
    ) -> List[Dict]:
        content = []
        current_text = ""
        parts = re.split(r'(<\|[^|]+\|>)', prompt)

        for part in parts:
            match = re.match(r'<\|([^|]+)\|>', part)
            if match:
                placeholder = match.group(1)
                if current_text.strip():
                    content.append({'type': 'text', 'text': current_text})
                    current_text = ""
                if placeholder in images:
                    content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:image/png;base64,{images[placeholder]}"
                        }
                    })
                else:
                    current_text += part
            else:
                current_text += part

        if current_text.strip():
            content.append({'type': 'text', 'text': current_text})

        return content

    def _call_vlm(
        self,
        prompt: str,
        images: Dict[str, str],
        max_tokens: int = 4096,
    ) -> Tuple[str, bool]:
        retry_times = 0
        max_retries = 5
        content = self._parse_prompt_with_images(prompt, images)
        messages = [{'role': 'user', 'content': content}]

        while retry_times < max_retries:
            try:
                self.logger.info(f'Calling visual model {self.visual_model}...')
                response = self.client.chat.completions.create(
                    model=self.visual_model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content, False
            except Exception as e:
                self.logger.error(f'VLM API call error: {type(e).__name__} - {str(e)}')
                retry_times += 1
                time.sleep(5)

        return None, True

    def compare_prototype(
        self,
        prototype_screenshot_b64: str,
        actual_screenshot_b64: str,
    ) -> List[Dict]:
        self.logger.info('Comparing prototype...')

        prompt = VISUAL_PROMPT.format(
            prototype_page="<|prototype_page|>",
            actual_page="<|actual_page|>",
        )
        images = {
            'prototype_page': prototype_screenshot_b64,
            'actual_page': actual_screenshot_b64,
        }

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            self.logger.info(f'Prototype comparison attempt {attempt}/{max_retries}')

            response, error = self._call_vlm(prompt=prompt, images=images)

            if error or not response:
                self.logger.error(f'Prototype comparison API call failed on attempt {attempt}')
                if attempt < max_retries:
                    continue
                return []

            self.logger.info(f'Prototype comparison response: {response}')

            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r'\[.*\]', response, re.DOTALL)
                    json_str = json_match.group(0) if json_match else response

                scores = json.loads(json_str)

                if not scores:
                    self.logger.warning(f'Empty result on attempt {attempt}')
                    if attempt < max_retries:
                        continue
                    return []

                self.logger.info(f'Parsed {len(scores)} component scores')
                return scores

            except json.JSONDecodeError as e:
                self.logger.error(f'JSON parse error on attempt {attempt}: {e}')
                if attempt < max_retries:
                    continue
                return []

        return []

    async def score_all_prototypes(
        self,
        workflow_item: Dict,
        workflow_idx: int,
        dataset_path: str,
        output_dir: Path,
        actual_pages: Dict[str, bytes],
    ) -> Dict[str, Any]:
        """Score all prototypes for a workflow item.

        Args:
            workflow_item: Workflow definition with 'prototype' field
            workflow_idx: Workflow index
            dataset_path: Path to dataset containing prototypes/ directory
            output_dir: Output directory (workspace)
            actual_pages: Dict mapping prototype names to PNG bytes,
                          e.g. {"homepage": bytes, "register_page": bytes}

        Returns:
            Dict mapping prototype names to score data
        """
        prototype_info = workflow_item.get('prototype', {})
        results = {}

        test_results_base = Path(output_dir) / 'test_results'
        prototypes_dir = test_results_base / 'prototypes'
        prototypes_dir.mkdir(parents=True, exist_ok=True)

        for proto_name, proto_config in prototype_info.items():
            try:
                self.logger.info(f'Processing prototype: {proto_name}')

                prototype_path = os.path.join(dataset_path, 'prototypes', f'{proto_name}.jpg')
                if not os.path.exists(prototype_path):
                    self.logger.error(f"Prototype not found: {prototype_path}")
                    continue

                prototype_b64 = base64.b64encode(
                    open(prototype_path, 'rb').read()
                ).decode('utf-8')

                actual_data = actual_pages.get(proto_name)
                if not actual_data:
                    self.logger.warning(f"No screenshot for prototype {proto_name}")
                    continue

                actual_b64 = base64.b64encode(actual_data).decode('utf-8')

                scores = self.compare_prototype(
                    prototype_screenshot_b64=prototype_b64,
                    actual_screenshot_b64=actual_b64,
                )
                results[proto_name] = scores

                scores_file = prototypes_dir / f'{proto_name}_scores.json'
                with open(scores_file, 'w', encoding='utf-8') as f:
                    json.dump(scores, f, indent=2, ensure_ascii=False)

            except Exception as e:
                self.logger.error(f"Failed to process prototype {proto_name}: {e}", exc_info=True)

        return results
