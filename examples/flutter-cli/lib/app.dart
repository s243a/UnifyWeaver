import 'package:flutter/material.dart';
import 'dart:math';

// Helper function for formatting file sizes
String formatSize(dynamic bytes) {
  if (bytes == null || bytes == 0) return '0 B';
  const sizes = ['B', 'KB', 'MB', 'GB'];
  final i = (bytes > 0) ? (log(bytes) / log(1024)).floor() : 0;
  return '\${(bytes / pow(1024, i)).toStringAsFixed(1)} \${sizes[i]}';
}

class App extends StatefulWidget {
  const App({super.key});

  @override
  State<App> createState() => _AppState();
}

class _AppState extends State<App> {
  // Loading and error state
  bool _loading = false;
  String _error = '';

  // Browse state
  final Map<String, dynamic> _browse = {
    'path': '.',
    'entries': <Map<String, dynamic>>[
      {'name': 'example.txt', 'type': 'file', 'size': 1024},
      {'name': 'folder', 'type': 'directory', 'size': 0},
    ],
    'selected': null,
    'parent': false,
  };

  // Working directory
  String _workingDir = '.';

  // Controllers
  final TextEditingController _pathController = TextEditingController();

  @override
  void dispose() {
    _pathController.dispose();
    super.dispose();
  }

  // Navigation handlers
  void navigateUp() {
    setState(() {
      final parts = (_browse['path'] as String).split('/');
      if (parts.length > 1) {
        parts.removeLast();
        _browse['path'] = parts.join('/');
        if (_browse['path'].isEmpty) _browse['path'] = '.';
      }
      _browse['parent'] = _browse['path'] != '.';
    });
    debugPrint('Navigate up to: \${_browse["path"]}');
  }

  void handleEntryClick(Map<String, dynamic> entry) {
    setState(() {
      if (entry['type'] == 'directory') {
        final currentPath = _browse['path'] as String;
        _browse['path'] = currentPath == '.'
            ? entry['name']
            : '\$currentPath/\${entry["name"]}';
        _browse['parent'] = true;
        _browse['selected'] = null;
      } else {
        _browse['selected'] = entry['name'];
      }
    });
  }

  void setWorkingDir() {
    setState(() {
      _workingDir = _browse['path'] as String;
    });
    debugPrint('Working dir set to: \$_workingDir');
  }

  void viewFile() {
    final selected = _browse['selected'];
    if (selected != null) {
      debugPrint('View file: \$selected');
    }
  }

  void downloadFile() {
    final selected = _browse['selected'];
    if (selected != null) {
      debugPrint('Download file: \$selected');
    }
  }

  void searchHere() {
    debugPrint('Search in: \${_browse["path"]}');
  }

  @override
  Widget build(BuildContext context) {
    return Container(
  padding: EdgeInsets.all(20),
  decoration: BoxDecoration(
    color: Color(0xFF16213e),
    borderRadius: BorderRadius.circular(5),
  ),
  child:   Column(
    mainAxisAlignment: MainAxisAlignment.start,
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      Wrap(
        spacing: 10,
        runSpacing: 10,
        children: [
          if (_browse['parent'] != null)           OutlinedButton(
            onPressed: () => navigateUp(),
            child: Text("⬆️ Up"),
          ),
          SizedBox(height: 10),
          Text("📁 "),
          SizedBox(height: 10),
          Container(
            padding: EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Color(0xFF1a1a2e),
              borderRadius: BorderRadius.circular(3),
            ),
            child: Text(_browse['path'].toString(), style: TextStyle(fontFamily: "monospace")),
          ),
          SizedBox(height: 10),
          ElevatedButton(
            onPressed: null,
            child: Text("📌 Set as Working Dir"),
          ),
        ],
      ),
      SizedBox(height: 15),
      if (_browse['entries'] != null)       Text(""),
      SizedBox(height: 15),
      ConstrainedBox(
        constraints: BoxConstraints(maxHeight: 400),
        child: SingleChildScrollView(
          child:         Column(
          children: _browse['entries'].map((entry) =>             Container(
              padding: EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Color(0xFF16213e),
                borderRadius: BorderRadius.circular(5),
              ),
              child:               Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.start,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Icon(Icons.circle, size: 24),
                      SizedBox(height: 8),
                      Text(entry['name'].toString()),
                    ],
                  ),
                  SizedBox(height: 0),
                  Text(formatSize(entry['size']), style: TextStyle(fontSize: 12, color: Colors.grey)),
                ],
              ),
            ),
).toList(),
        ),
        ),
      ),
      SizedBox(height: 15),
      if (((_browse['entries'] == null || _browse['entries'].isEmpty) && !_loading))       Text("Empty directory", style: TextStyle(color: Colors.grey)),
      SizedBox(height: 15),
      if (_browse['selected'] != null)       Container(
        padding: EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Color(0xFF16213e),
          borderRadius: BorderRadius.circular(5),
        ),
        child:         Column(
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text("Selected file:", style: TextStyle(fontSize: 12, color: Colors.grey)),
            SizedBox(height: 10),
            Container(
              padding: EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Color(0xFF1a1a2e),
                borderRadius: BorderRadius.circular(3),
              ),
              child: Text(_browse['selected'].toString(), style: TextStyle(fontFamily: "monospace")),
            ),
            SizedBox(height: 10),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [
                ElevatedButton(
                  onPressed: () => viewFile(),
                  child: Text("View Contents"),
                ),
                SizedBox(height: 10),
                ElevatedButton(
                  onPressed: () => downloadFile(),
                  child: Text("📥 Download"),
                ),
                SizedBox(height: 10),
                OutlinedButton(
                  onPressed: () => searchHere(),
                  child: Text("Search Here"),
                ),
              ],
            ),
          ],
        ),
      ),
    ],
  ),
),

  }
}
