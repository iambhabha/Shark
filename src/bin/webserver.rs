//! A tiny local web server that lets you play against Mythos in your browser.
//!
//! It uses ONLY the standard library (a minimal HTTP/1.1 server) plus the
//! `mythos` engine crate, so there are no external dependencies. Run it, open
//! http://localhost:8080, and play.
//!
//!   cargo run --release --bin webserver            # port 8080
//!   cargo run --release --bin webserver -- 9000    # custom port
//!
//! The single API endpoint is `POST /api/play`. The frontend (served at `/`)
//! sends the full move history; this server is the referee (via the crate's
//! legal-move generator) AND the opponent (via the search). It is stateless:
//! every request rebuilds the position from the move list.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use mythos::movegen::generate_legal;
use mythos::position::Position;
use mythos::search::{SearchLimits, Searcher};
use mythos::types::{Color, Square};

/// The browser page, baked into the binary at compile time.
const INDEX_HTML: &str = include_str!("../../web/index.html");

fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Could not bind to port {port}: {e}");
            eprintln!("Try another port, e.g. `webserver 9000`.");
            std::process::exit(1);
        }
    };

    // One shared searcher keeps its transposition table warm across moves.
    let searcher = Arc::new(Mutex::new(Searcher::new(64)));

    // Use NNUE for the browser game when a net (`mythos.nnue`) is present; else the
    // hand-crafted evaluation. Report which one is in effect.
    if let Some(net) = mythos::nnue::load_default() {
        searcher.lock().unwrap().set_net(Some(net));
        println!("NNUE evaluation loaded.");
    } else {
        println!("No NNUE net found; using hand-crafted evaluation.");
    }

    println!("Mythos web UI running.  Open  ->  http://localhost:{port}");
    println!("(Press Ctrl+C to stop.)");

    for stream in listener.incoming() {
        match stream {
            Ok(s) => handle(s, &searcher),
            Err(_) => continue,
        }
    }
}

/// Read one HTTP request, route it, and write the response. One request per
/// connection (Connection: close) — simple and plenty for a single local user.
fn handle(mut stream: TcpStream, searcher: &Arc<Mutex<Searcher>>) {
    let peek = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut reader = BufReader::new(peek);

    let mut request_line = String::new();
    if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
        return;
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    // Consume headers; note Content-Length so we can read the body.
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok();
    }
    let body = String::from_utf8_lossy(&body).to_string();

    let (status, ctype, payload) = route(&method, &path, &body, searcher);

    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream.write_all(header.as_bytes()).ok();
    stream.write_all(payload.as_bytes()).ok();
    stream.flush().ok();
}

/// Map (method, path) to a response: (status line, content-type, body).
fn route(
    method: &str,
    path: &str,
    body: &str,
    searcher: &Arc<Mutex<Searcher>>,
) -> (&'static str, &'static str, String) {
    match (method, path) {
        ("GET", "/") => ("200 OK", "text/html; charset=utf-8", INDEX_HTML.to_string()),
        ("GET", "/favicon.ico") => ("204 No Content", "text/plain", String::new()),
        ("POST", "/api/play") => (
            "200 OK",
            "application/json",
            api_play(body, searcher),
        ),
        _ => ("404 Not Found", "text/plain", "not found".to_string()),
    }
}

/// The one game endpoint. Rebuilds the position from the move history, optionally
/// lets Mythos reply, and returns the full board state as JSON.
fn api_play(body: &str, searcher: &Arc<Mutex<Searcher>>) -> String {
    let form = parse_form(body);

    let moves: Vec<String> = form
        .get("moves")
        .map(|s| {
            s.split(',')
                .filter(|x| !x.is_empty())
                .map(|x| x.to_string())
                .collect()
        })
        .unwrap_or_default();
    let movetime: u64 = form
        .get("movetime")
        .and_then(|s| s.parse().ok())
        .unwrap_or(500)
        .clamp(50, 5000);
    let want_engine = form.get("engine").map(|s| s == "1").unwrap_or(false);

    // Rebuild and validate the game so far.
    let (mut pos, mut keys) = match build_position(&moves) {
        Ok(v) => v,
        Err(e) => return format!("{{\"ok\":false,\"error\":\"{}\"}}", escape(&e)),
    };

    // Let Mythos move if asked and the game is still going.
    let mut engine_move: Option<String> = None;
    if want_engine {
        let (status, _, _, _) = finalize(&mut pos, &keys);
        if status == "ongoing" {
            let stop = Arc::new(AtomicBool::new(false));
            let limits = SearchLimits {
                movetime_ms: Some(movetime),
                ..Default::default()
            };
            let result = {
                let mut s = searcher.lock().unwrap();
                s.search(&pos, &limits, &stop)
            };
            let m = result.best_move;
            if !m.is_none() {
                engine_move = Some(m.to_string());
                pos.make_move(m);
                keys.push(pos.key());
            }
        }
    }

    let (status, winner, legal, check) = finalize(&mut pos, &keys);

    let legal_json = legal
        .iter()
        .map(|m| format!("\"{m}\""))
        .collect::<Vec<_>>()
        .join(",");
    let winner_json = match winner {
        Some(w) => format!("\"{w}\""),
        None => "null".to_string(),
    };
    let engine_json = match &engine_move {
        Some(m) => format!("\"{m}\""),
        None => "null".to_string(),
    };
    let side = if pos.side_to_move() == Color::White {
        "white"
    } else {
        "black"
    };

    format!(
        "{{\"ok\":true,\"board\":\"{}\",\"fen\":\"{}\",\"side\":\"{}\",\"status\":\"{}\",\"winner\":{},\"engineMove\":{},\"legal\":[{}],\"check\":{}}}",
        board_string(&pos),
        escape(&pos.to_fen()),
        side,
        status,
        winner_json,
        engine_json,
        legal_json,
        check
    )
}

/// Rebuild a Position by replaying UCI moves from the start, validating each.
/// Returns the position and the list of Zobrist keys (for repetition checks).
fn build_position(moves: &[String]) -> Result<(Position, Vec<u64>), String> {
    let mut pos = Position::startpos();
    let mut keys = vec![pos.key()];
    for (i, mv) in moves.iter().enumerate() {
        let list = generate_legal(&mut pos);
        let mut chosen = None;
        for m in &list {
            if m.to_string() == *mv {
                chosen = Some(m);
                break;
            }
        }
        match chosen {
            Some(m) => {
                pos.make_move(m);
                keys.push(pos.key());
            }
            None => return Err(format!("illegal move '{mv}' at ply {}", i + 1)),
        }
    }
    Ok((pos, keys))
}

/// Determine game status + legal moves + check for the current position.
fn finalize(
    pos: &mut Position,
    keys: &[u64],
) -> (&'static str, Option<&'static str>, Vec<String>, bool) {
    let list = generate_legal(pos);
    let legal: Vec<String> = (&list).into_iter().map(|m| m.to_string()).collect();
    let check = pos.in_check();

    let status_winner: (&'static str, Option<&'static str>) = if legal.is_empty() {
        if check {
            let winner = if pos.side_to_move() == Color::White {
                "black"
            } else {
                "white"
            };
            ("checkmate", Some(winner))
        } else {
            ("stalemate", None)
        }
    } else if pos.halfmove_clock() >= 100 {
        ("draw", None)
    } else {
        let k = *keys.last().unwrap();
        if keys.iter().filter(|&&x| x == k).count() >= 3 {
            ("draw", None)
        } else {
            ("ongoing", None)
        }
    };

    (status_winner.0, status_winner.1, legal, check)
}

/// 64-char board string, index 0 = a1 .. 63 = h8, FEN letters, '.' = empty.
fn board_string(pos: &Position) -> String {
    let mut s = String::with_capacity(64);
    for i in 0..64 {
        let sq = Square::from_index(i).unwrap();
        match pos.piece_at(sq) {
            Some(p) => s.push(p.to_char()),
            None => s.push('.'),
        }
    }
    s
}

/// Parse an `a=b&c=d` form body into a map.
fn parse_form(body: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for pair in body.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            m.insert(k.to_string(), url_decode(v));
        }
    }
    m
}

/// Minimal percent-decoding (the frontend may `encodeURIComponent` the moves).
fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let h = u8::from_str_radix(&s[i + 1..i + 3], 16);
                match h {
                    Ok(b) => {
                        out.push(b);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

/// Escape a string for embedding inside a JSON string literal.
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
