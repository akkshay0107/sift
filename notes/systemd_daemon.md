# Running the Indexer Daemon via Systemd

To run the Sift Indexer daemon continuously in the background using `systemd` (user-level, no root or sudo required), follow these steps:

1. Create the systemd service directory if it doesn't exist:

```bash
mkdir -p ~/.config/systemd/user/
```

2. Create a service file using your preferred editor:

```bash
nano ~/.config/systemd/user/sift-indexer.service
```

3. Paste the following configuration into the file:
   _(Make sure to double check that the path to `uv` is correct. You can find your path by running `which uv`)_

```ini
[Unit]
Description=Sift Background Indexer Daemon
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/akkshaysr/main/sift
ExecStart=/home/akkshaysr/.cargo/bin/uv run python -m src.indexer.daemon
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

4. Reload the systemd user daemon so it registers the new file:

```bash
systemctl --user daemon-reload
```

5. Enable the service to start automatically on system boot:

```bash
systemctl --user enable sift-indexer.service
```

6. Start the service now for the first time:

```bash
systemctl --user start sift-indexer.service
```

### Useful Commands

Check if it's running:

```bash
systemctl --user status sift-indexer.service
```

Tail the logs live to make sure watchdog is indexing items:

```bash
journalctl --user -u sift-indexer.service -f
```

Stop the daemon manually:

```bash
systemctl --user stop sift-indexer.service
```
