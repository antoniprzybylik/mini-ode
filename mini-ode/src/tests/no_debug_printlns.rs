use std::{fs, path::Path};

use syn::{
    Macro,
    spanned::Spanned,
    visit::{self, Visit},
};

struct PrintlnFinder {
    occurrences: Vec<(usize, usize)>,
}

impl<'ast> Visit<'ast> for PrintlnFinder {
    fn visit_macro(&mut self, mac: &'ast Macro) {
        if mac.path.is_ident("println") {
            let start = mac.span().start();
            self.occurrences.push((start.line, start.column + 1));
        }

        visit::visit_macro(self, mac);
    }
}

#[test]
fn no_debug_printlns() {
    fn visit_dir(dir: &Path, offenders: &mut Vec<String>) {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.is_dir() {
                visit_dir(&path, offenders);
                continue;
            }

            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }

            let src = fs::read_to_string(&path).unwrap();

            let file = syn::parse_file(&src)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));

            let mut finder = PrintlnFinder {
                occurrences: Vec::new(),
            };
            finder.visit_file(&file);

            for (line, col) in finder.occurrences {
                offenders.push(format!("{}:{}:{}", path.display(), line, col));
            }
        }
    }

    let mut offenders = Vec::new();
    visit_dir(Path::new("src"), &mut offenders);

    assert!(
        offenders.is_empty(),
        "Found forbidden `println!` macro(s):\n{}",
        offenders.join("\n"),
    );
}
