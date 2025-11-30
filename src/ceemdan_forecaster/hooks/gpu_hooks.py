# -*- coding: utf-8 -*-
"""GPU memory management hooks."""
import gc
import logging

import torch
from kedro.framework.hooks import hook_impl

logger = logging.getLogger(__name__)


class GPUMemoryHook:
    """Clean up GPU memory after each node to prevent OOM errors.

    This hook ensures that CUDA memory is properly released between
    node executions, which is important when training many models
    (450 models in production configuration).
    """

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs):
        """Clean up GPU memory after each node completes.

        Args:
            node: The node that just finished running
            catalog: The data catalog
            inputs: The inputs to the node
            outputs: The outputs from the node
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug(
                f"Cleaned GPU memory after node: {node.name}"
            )

    @hook_impl
    def on_node_error(self, error, node, catalog, inputs):
        """Clean up GPU memory when a node fails.

        Args:
            error: The exception that was raised
            node: The node that failed
            catalog: The data catalog
            inputs: The inputs to the node
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.warning(
                f"Cleaned GPU memory after error in node: {node.name}"
            )
