"""
Test suite for the CLI module.

Tests command-line interface functionality for training, classification,
batch processing, and configuration management.
"""

import os
import pytest
import tempfile
import json
import csv
from pathlib import Path
from click.testing import CliRunner
from cli import (
    main,
    train,
    classify,
    batch_classify,
    analyze,
    config,
    validate_audio_dir,
    get_label_mapping,
    format_metrics,
)


class TestCLIHelpers:
    """Test CLI helper functions."""

    def test_get_label_mapping(self):
        """Test label mapping creation."""
        classes = ['AUV', 'Torpedo', 'Vessel']
        mapping = get_label_mapping(classes)
        
        assert isinstance(mapping, dict)
        assert len(mapping) == 3
        assert mapping['AUV'] == 0
        assert mapping['Torpedo'] == 1
        assert mapping['Vessel'] == 2

    def test_format_metrics(self):
        """Test metrics formatting."""
        metrics = {
            'accuracy': 0.95,
            'precision': 0.93,
            'recall': 0.91,
            'f1': 0.92,
        }
        formatted = format_metrics(metrics)
        
        assert isinstance(formatted, str)
        assert 'accuracy' in formatted
        assert '0.95' in formatted

    def test_validate_audio_dir_with_subdirs(self, tmp_path):
        """Test audio directory validation with subdirectories."""
        # Create test directory structure
        class1_dir = tmp_path / "AUV"
        class2_dir = tmp_path / "Torpedo"
        class1_dir.mkdir()
        class2_dir.mkdir()

        # Create dummy WAV files
        (class1_dir / "sample1.wav").touch()
        (class2_dir / "sample2.wav").touch()

        result = validate_audio_dir(str(tmp_path))
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert 'AUV' in result
        assert 'Torpedo' in result

    def test_validate_audio_dir_empty_raises(self, tmp_path):
        """Test validation raises on empty directory."""
        with pytest.raises(ValueError):
            validate_audio_dir(str(tmp_path))

    def test_validate_audio_dir_no_subdirs_raises(self, tmp_path):
        """Test validation raises when no subdirectories exist."""
        (tmp_path / "file.txt").touch()
        
        with pytest.raises(ValueError, match="No subdirectories"):
            validate_audio_dir(str(tmp_path))

    def test_validate_audio_dir_nonexistent_raises(self):
        """Test validation raises for nonexistent directory."""
        with pytest.raises(ValueError, match="does not exist"):
            validate_audio_dir("/nonexistent/path")


class TestCLIMainCommand:
    """Test main CLI group."""

    def test_main_help(self):
        """Test main command help output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert 'Commands:' in result.output or 'train' in result.output

    def test_main_version(self):
        """Test version flag."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert '1.0.0' in result.output


class TestTrainCommand:
    """Test train subcommand."""

    def test_train_help(self):
        """Test train command help."""
        runner = CliRunner()
        result = runner.invoke(train, ['--help'])
        
        assert result.exit_code == 0
        assert 'data-dir' in result.output
        assert 'model-type' in result.output

    def test_train_missing_required_args(self):
        """Test train fails without required arguments."""
        runner = CliRunner()
        result = runner.invoke(train, [])
        
        assert result.exit_code != 0
        assert 'data-dir' in result.output or 'Error' in result.output

    def test_train_with_nonexistent_dir(self):
        """Test train fails with nonexistent data directory."""
        runner = CliRunner()
        result = runner.invoke(train, [
            '--data-dir', '/nonexistent/path'
        ])
        
        assert result.exit_code != 0


class TestClassifyCommand:
    """Test classify subcommand."""

    def test_classify_help(self):
        """Test classify command help."""
        runner = CliRunner()
        result = runner.invoke(classify, ['--help'])
        
        assert result.exit_code == 0
        assert 'audio-file' in result.output
        assert 'model' in result.output

    def test_classify_missing_required_args(self):
        """Test classify fails without required arguments."""
        runner = CliRunner()
        result = runner.invoke(classify, [])
        
        assert result.exit_code != 0


class TestBatchClassifyCommand:
    """Test batch-classify subcommand."""

    def test_batch_classify_help(self):
        """Test batch-classify command help."""
        runner = CliRunner()
        result = runner.invoke(batch_classify, ['--help'])
        
        assert result.exit_code == 0
        assert 'input-dir' in result.output
        assert 'model' in result.output

    def test_batch_classify_missing_required_args(self):
        """Test batch-classify fails without required arguments."""
        runner = CliRunner()
        result = runner.invoke(batch_classify, [])
        
        assert result.exit_code != 0


class TestAnalyzeCommand:
    """Test analyze subcommand."""

    def test_analyze_help(self):
        """Test analyze command help."""
        runner = CliRunner()
        result = runner.invoke(analyze, ['--help'])
        
        assert result.exit_code == 0
        assert 'audio-file' in result.output

    def test_analyze_missing_required_args(self):
        """Test analyze fails without required arguments."""
        runner = CliRunner()
        result = runner.invoke(analyze, [])
        
        assert result.exit_code != 0


class TestConfigCommand:
    """Test config subcommand."""

    def test_config_help(self):
        """Test config command help."""
        runner = CliRunner()
        result = runner.invoke(config, ['--help'])
        
        assert result.exit_code == 0
        assert 'show' in result.output or 'generate' in result.output

    def test_config_show(self):
        """Test config --show."""
        runner = CliRunner()
        result = runner.invoke(config, ['--show'])
        
        assert result.exit_code == 0
        assert 'preprocessor' in result.output or 'nmf' in result.output

    def test_config_generate(self, tmp_path):
        """Test config --generate."""
        config_file = str(tmp_path / "test_config.yaml")
        runner = CliRunner()
        result = runner.invoke(config, ['--generate', config_file])
        
        assert result.exit_code == 0
        assert os.path.exists(config_file)

    def test_config_no_args_fails(self):
        """Test config fails without arguments."""
        runner = CliRunner()
        result = runner.invoke(config, [])
        
        assert result.exit_code != 0
