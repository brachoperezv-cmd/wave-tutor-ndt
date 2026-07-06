# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from unittest.mock import patch

import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import PIVerdict, root_agent


class MockGenerateContentResponse:
    def __init__(self, text):
        self.text = text


# Mock implementation of AsyncModels.generate_content for local offline testing
async def mock_generate_content(self, model, contents, config=None, **kwargs):
    # Check if this is the PI Audit call
    if config and getattr(config, "response_schema", None) == PIVerdict:
        return MockGenerateContentResponse(
            json.dumps(
                {
                    "approved": True,
                    "feedback_to_ra": "Mock PI approval feedback.",
                }
            )
        )
    else:
        # RA draft explanation call
        return MockGenerateContentResponse("Mock RA draft lesson explanation.")


@pytest.mark.asyncio
async def test_agent_practice_mode_pass() -> None:
    session_service = InMemorySessionService()
    session = await session_service.create_session(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    # Lead correct speeds are: cL around 2050, cS around 700
    # Thickness = 20mm
    mock_info = {
        "material": "lead",
        "thickness_mm": 20.0,
        "frequency_khz": 1000.0,
        "actual_cL_m_s": 2050.0,
        "actual_cS_m_s": 700.0,
        "longitudinal_delay_us": 9.756,
        "shear_delay_us": 28.571,
        "excit_file": "outputs/lead_excit.txt",
        "longi_file": "outputs/lead_longi.txt",
        "shear_file": "outputs/lead_shear.txt",
        "artifact_info": {},
    }

    orig_exists = os.path.exists

    def mock_exists(path):
        if str(path).endswith("_meta.json"):
            return False
        return orig_exists(path)

    with (
        patch("app.agent.generate_signal_data", return_value=mock_info),
        patch(
            "app.agent.find_signal_peaks",
            return_value=[0.417, 1.267, 2.25, 3.25, 4.233, 12.006, 30.821],
        ),
        patch(
            "app.agent.find_correlation_peaks",
            side_effect=lambda f1, f2: [9.756] if "longi" in f2 else [28.571],
        ),
        patch("os.path.exists", side_effect=mock_exists),
        patch(
            "google.genai.models.AsyncModels.generate_content",
            new=mock_generate_content,
        ),
    ):
        prompt = "practice material=lead thickness=20mm frequency=1000kHz l_ex=2.25 l_rec=12.006 s_ex=2.25 s_rec=30.821"
        message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

        events = list(
            runner.run(
                new_message=message,
                user_id="test_user",
                session_id=session.id,
            )
        )

    assert len(events) > 0
    updated_session = await session_service.get_session(
        session_id=session.id, app_name="test", user_id="test_user"
    )
    assert updated_session.state.get("practice_mode") is True
    assert updated_session.state.get("practice_l_ex") == 2.25
    assert updated_session.state.get("practice_l_rec") == 12.006
    assert updated_session.state.get("practice_s_ex") == 2.25
    assert updated_session.state.get("practice_s_rec") == 30.821

    # Check history contains a PASS verdict
    history = updated_session.state.get("error_history", [])
    assert len(history) > 0
    assert history[-1]["error_type"] == "PASS"


@pytest.mark.asyncio
async def test_agent_practice_mode_fail() -> None:
    session_service = InMemorySessionService()
    session = await session_service.create_session(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    mock_info = {
        "material": "lead",
        "thickness_mm": 20.0,
        "frequency_khz": 1000.0,
        "actual_cL_m_s": 2050.0,
        "actual_cS_m_s": 700.0,
        "longitudinal_delay_us": 9.756,
        "shear_delay_us": 28.571,
        "excit_file": "outputs/lead_excit.txt",
        "longi_file": "outputs/lead_longi.txt",
        "shear_file": "outputs/lead_shear.txt",
        "artifact_info": {},
    }

    orig_exists = os.path.exists

    def mock_exists(path):
        if str(path).endswith("_meta.json"):
            return False
        return orig_exists(path)

    with (
        patch("app.agent.generate_signal_data", return_value=mock_info),
        patch(
            "app.agent.find_signal_peaks",
            return_value=[0.417, 1.267, 2.25, 2.75, 3.25, 4.233, 30.821],
        ),
        patch(
            "app.agent.find_correlation_peaks",
            side_effect=lambda f1, f2: [9.756] if "longi" in f2 else [28.571],
        ),
        patch("os.path.exists", side_effect=mock_exists),
        patch(
            "google.genai.models.AsyncModels.generate_content",
            new=mock_generate_content,
        ),
    ):
        # Impossibly fast L value (Category 2 Error)
        prompt = "practice material=lead thickness=20mm frequency=1000kHz l_ex=2.25 l_rec=2.75 s_ex=2.25 s_rec=30.821"
        message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

        events = list(
            runner.run(
                new_message=message,
                user_id="test_user",
                session_id=session.id,
            )
        )

    assert len(events) > 0
    updated_session = await session_service.get_session(
        session_id=session.id, app_name="test", user_id="test_user"
    )
    assert updated_session.state.get("practice_mode") is True

    # Check history contains a REJECT_CAT2 verdict
    history = updated_session.state.get("error_history", [])
    assert len(history) > 0
    assert history[-1]["error_type"] == "REJECT_CAT2"
