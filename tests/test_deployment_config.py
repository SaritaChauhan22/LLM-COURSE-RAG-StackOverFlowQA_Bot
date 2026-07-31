import tomllib
import unittest
from pathlib import Path


class DeploymentConfigTests(unittest.TestCase):
    def test_streamlit_public_access_config_exists(self):
        config_path = Path(".streamlit/config.toml")
        self.assertTrue(config_path.exists(), "Streamlit deployment config is missing")

        with config_path.open("rb") as handle:
            config = tomllib.load(handle)

        server_config = config["server"]
        self.assertEqual(server_config["address"], "0.0.0.0")
        self.assertTrue(server_config["headless"])


if __name__ == "__main__":
    unittest.main()
