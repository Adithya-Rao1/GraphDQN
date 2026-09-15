import os
from typing import Any, Optional

import requests
import streamlit as st

API_BASE_URL = os.environ.get("GRAPHDQN_API_URL", "http://localhost:8000")


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class ApiClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict:
        token = st.session_state.get("token")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _request(self, method: str, path: str, **kwargs) -> Any:
        response = requests.request(method, f"{self.base_url}{path}", headers=self._headers(), timeout=30, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, str(detail))
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def signup(self, email: str, password: str) -> dict:
        return self._request("POST", "/api/auth/signup", json={"email": email, "password": password})

    def login(self, email: str, password: str) -> dict:
        return self._request("POST", "/api/auth/login", json={"email": email, "password": password})

    def me(self) -> dict:
        return self._request("GET", "/api/auth/me")

    def admet_properties(self) -> list:
        return self._request("GET", "/api/meta/admet-properties")

    def example_targets(self) -> list:
        return self._request("GET", "/api/meta/example-targets")

    def preview_molecule(self, smiles: str) -> dict:
        return self._request("POST", "/api/molecules/preview", json={"smiles": smiles})

    def create_molecule(self, smiles: str, label: Optional[str] = None) -> dict:
        return self._request("POST", "/api/molecules", json={"smiles": smiles, "label": label})

    def list_molecules(self) -> list:
        return self._request("GET", "/api/molecules")

    def delete_molecule(self, molecule_id: int) -> None:
        self._request("DELETE", f"/api/molecules/{molecule_id}")

    def create_protein(self, name: str, sequence: str) -> dict:
        return self._request("POST", "/api/proteins", json={"name": name, "sequence": sequence})

    def list_proteins(self) -> list:
        return self._request("GET", "/api/proteins")

    def delete_protein(self, protein_id: int) -> None:
        self._request("DELETE", f"/api/proteins/{protein_id}")

    def create_config(self, payload: dict) -> dict:
        return self._request("POST", "/api/configs", json=payload)

    def list_configs(self) -> list:
        return self._request("GET", "/api/configs")

    def delete_config(self, config_id: int) -> None:
        self._request("DELETE", f"/api/configs/{config_id}")

    def start_run(self, payload: dict) -> dict:
        return self._request("POST", "/api/runs", json=payload)

    def list_runs(self, config_id: Optional[int] = None) -> list:
        params = {"config_id": config_id} if config_id is not None else {}
        return self._request("GET", "/api/runs", params=params)

    def get_run(self, run_id: int) -> dict:
        return self._request("GET", f"/api/runs/{run_id}")

    def cancel_run(self, run_id: int) -> dict:
        return self._request("POST", f"/api/runs/{run_id}/cancel")

    def finetune_run(self, run_id: int, payload: dict) -> dict:
        return self._request("POST", f"/api/runs/{run_id}/finetune", json=payload)

    def generate_candidates(self, run_id: int, payload: dict) -> dict:
        return self._request("POST", f"/api/runs/{run_id}/generate", json=payload)

    def get_generation_batch(self, batch_id: int) -> dict:
        return self._request("GET", f"/api/generation-batches/{batch_id}")

    def list_candidates(self, generation_batch_id: Optional[int] = None, training_run_id: Optional[int] = None,
                         target_protein_id: Optional[int] = None, config_id: Optional[int] = None) -> list:
        params = {}
        if generation_batch_id is not None:
            params["generation_batch_id"] = generation_batch_id
        if training_run_id is not None:
            params["training_run_id"] = training_run_id
        if target_protein_id is not None:
            params["target_protein_id"] = target_protein_id
        if config_id is not None:
            params["config_id"] = config_id
        return self._request("GET", "/api/candidates", params=params)

    def score_candidate(self, candidate_id: int, rating: float, notes: Optional[str] = None) -> dict:
        return self._request("POST", f"/api/candidates/{candidate_id}/score",
                              json={"rating": rating, "notes": notes})


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()
