custom_hooks = [
    dict(
        type='EMAIterHook',
        level=30,  # NORMAL
    ),  # level type : PriorityStatus, str, int
    dict(
        type='EMAEpochHook',
        level=40,
    ),
]
