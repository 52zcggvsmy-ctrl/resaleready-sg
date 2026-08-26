"""Tests for the prototype authentication boundary."""

import unittest
from unittest.mock import patch

import src.auth as auth


class AuthTests(unittest.TestCase):
    def test_auth_credentials_are_loaded_from_streamlit_secrets(self) -> None:
        secrets = {"auth": {"username": " admin ", "password": "new-password"}}
        with patch.object(auth.st, "secrets", secrets):
            self.assertEqual(auth.get_auth_credentials(), ("admin", "new-password"))

    def test_auth_configuration_fails_closed_when_missing_or_malformed(self) -> None:
        invalid_settings = (
            {},
            {"auth": {}},
            {"auth": {"username": "admin", "password": ""}},
            {"auth": {"username": "admin", "password": 1234}},
        )
        for secrets in invalid_settings:
            with self.subTest(secrets=secrets), patch.object(auth.st, "secrets", secrets):
                self.assertIsNone(auth.get_auth_credentials())

    def test_credentials_match_only_when_both_values_match(self) -> None:
        self.assertTrue(
            auth.credentials_match("admin", "new-password", "admin", "new-password")
        )
        self.assertFalse(
            auth.credentials_match("wrong", "new-password", "admin", "new-password")
        )
        self.assertFalse(
            auth.credentials_match("admin", "wrong", "admin", "new-password")
        )

    def test_credentials_match_supports_unicode_without_error(self) -> None:
        self.assertTrue(
            auth.credentials_match("buyer", "住宅-pass", "buyer", "住宅-pass")
        )


if __name__ == "__main__":
    unittest.main()
