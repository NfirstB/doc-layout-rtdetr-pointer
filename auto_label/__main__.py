"""
Auto Label Pipeline — Entry Point

Usage:
    python -m auto_label.pipeline /path/to/paper.pdf --page 0 --dpi 150 -o ./outputs
"""

if __name__ == "__main__":
    from auto_label.pipeline import main
    main()
