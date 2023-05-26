mod to_jit;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn to_jit(attr: TokenStream, item: TokenStream) -> TokenStream {
    to_jit::to_jit_inner(attr, item)
}