# =============================================================================
#  Compatibility entry for Figure 4
#
#  The current Figure 4 requested in the paper is the ST330IR001 four-panel
#  anomaly-detection visualization, not an attention heatmap. The real template
#  lives in:
#    scripts/research_paper/fig4_st330ir001_visualization_template.py
#
#  You may still run this file; it forwards to the current Figure 4 template.
# =============================================================================

from fig4_st330ir001_visualization_template import main


if __name__ == "__main__":
    main()
