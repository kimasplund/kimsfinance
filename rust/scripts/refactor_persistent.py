#!/usr/bin/env python3
"""
Refactor persistent/mod.rs to make it generic over PersistentIndicator trait.

This script performs the following transformations:
1. Make TaskBatch generic over I: PersistentIndicator
2. Make all helper functions generic
3. Replace ROC-specific kernel compilation with trait-based compilation
"""

import re
import sys

def refactor_file(input_path, output_path):
    with open(input_path, 'r') as f:
        content = f.read()

    # 1. Fix function signatures: Add <I: PersistentIndicator> to allocate_batch_buffers
    content = re.sub(
        r'fn allocate_batch_buffers\(device: &GpuDevice, batch: &TaskBatch\)',
        r'fn allocate_batch_buffers<I: PersistentIndicator>(device: &GpuDevice, batch: &TaskBatch<I>)',
        content
    )

    # 2. Fix function signatures: Add <I: PersistentIndicator> to upload_batch_data
    content = re.sub(
        r'fn upload_batch_data\(\s*device: &GpuDevice,\s*batch: &TaskBatch,',
        r'fn upload_batch_data<I: PersistentIndicator>(\n    device: &GpuDevice,\n    batch: &TaskBatch<I>,',
        content
    )

    # 3. Fix execute_batch signature
    content = re.sub(
        r'pub fn execute_batch\(device: &GpuDevice, batch: &TaskBatch\)',
        r'pub fn execute_batch<I: PersistentIndicator>(device: &GpuDevice, batch: &TaskBatch<I>)',
        content
    )

    # 4. Replace compile_persistent_kernel call with I::compile_kernel
    content = re.sub(
        r'let func = compile_persistent_kernel\(&self\._device\)\?;',
        r'let func = I::compile_kernel(&self._device)?;',
        content
    )

    # 5. Fix batch.sizes references
    content = re.sub(
        r'for &size in &batch\.sizes',
        r'for size in batch.tasks().iter().map(|t| t.data.len() as i32)',
        content
    )

    # 6. Fix batch.inputs references
    content = re.sub(
        r'batch\.inputs\.iter\(\)',
        r'batch.tasks().iter().map(|t| &t.data)',
        content
    )

    # 7. Fix batch.periods references (for i32 params)
    content = re.sub(
        r'let d_periods = device\.copy_to_device_i32\(&batch\.periods\)\?;',
        r'''// Extract parameters (generic - works for Copy types)
    let periods: Vec<i32> = batch.tasks()
        .iter()
        .map(|t| unsafe { std::mem::transmute_copy::<I::Params, i32>(&t.params) })
        .collect();
    let d_periods = device.copy_to_device_i32(&periods)?;''',
        content
    )

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"✓ Refactored {input_path} -> {output_path}")

if __name__ == '__main__':
    refactor_file(
        'src/gpu/persistent/mod.rs',
        'src/gpu/persistent/mod.rs'
    )
