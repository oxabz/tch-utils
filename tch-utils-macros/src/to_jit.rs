use darling::{ast::NestedMeta, Error, FromMeta};
use proc_macro::{TokenStream};
use proc_macro2::{Ident, Span};
use syn::{spanned::Spanned, token::Pub, ItemFn, Signature};

#[derive(Debug, FromMeta)]
enum Kind {
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,
}

impl Default for Kind {
    fn default() -> Self {
        Kind::Float
    }
}

impl Kind{
    fn to_ident(&self)->Ident{
        match self {
            Kind::Uint8 => Ident::new("Uint8", Span::call_site()),
            Kind::Int8 => Ident::new("Int8", Span::call_site()),
            Kind::Int16 => Ident::new("Int16", Span::call_site()),
            Kind::Int => Ident::new("Int", Span::call_site()),
            Kind::Int64 => Ident::new("Int64", Span::call_site()),
            Kind::Half => Ident::new("Half", Span::call_site()),
            Kind::Float => Ident::new("Float", Span::call_site()),
            Kind::Double => Ident::new("Double", Span::call_site()),
            Kind::ComplexHalf => Ident::new("ComplexHalf", Span::call_site()),
            Kind::ComplexFloat => Ident::new("ComplexFloat", Span::call_site()),
            Kind::ComplexDouble => Ident::new("ComplexDouble", Span::call_site()),
            Kind::Bool => Ident::new("Bool", Span::call_site()),
            Kind::QInt8 => Ident::new("QInt8", Span::call_site()),
            Kind::QUInt8 => Ident::new("QUInt8", Span::call_site()),
            Kind::QInt32 => Ident::new("QInt32", Span::call_site()),
            Kind::BFloat16 => Ident::new("BFloat16", Span::call_site()),
        }
    }
}

#[derive(Default, FromMeta)]
pub(crate) struct ToJitArgs {
    #[darling(default)]
    kind: Kind,
    tensor_size: Vec<usize>,
}

fn is_tensor(typ: &syn::Type) -> bool {
    match typ {
        syn::Type::Path(path) => path
            .path
            .segments
            .last()
            .map(|last| last.ident == syn::Ident::new("Tensor", Span::call_site()))
            .unwrap_or(false),
        _ => false,
    }
}

fn bad_type_error(span: Span) -> TokenStream {
    syn::Error::new(span, "The macro only suport (&Tensor)->Tensor functions")
        .to_compile_error()
        .into()
}

fn sig_check(func: &ItemFn) -> Result<(), TokenStream> {
    // Check that the function is a Module type function (&Tensor)->Tensor
    let Signature {
        ident,
        inputs,
        output,
        ..
    } = &func.sig;

    match output {
        syn::ReturnType::Type(_, typ) if is_tensor(&typ) => {}
        _ => {
            return Err(bad_type_error(ident.span()));
        }
    }

    if inputs.len() != 1 {
        return Err(bad_type_error(inputs.span()));
    }

    let input = &inputs[0];
    match input {
        syn::FnArg::Typed(pat_type) => {
            let typ = &pat_type.ty;
            match typ.as_ref() {
                syn::Type::Reference(ref_) => {
                    if ref_.mutability.is_some() {
                        return Err(bad_type_error(ref_.span()));
                    }
                    if !is_tensor(&ref_.elem) {
                        return Err(bad_type_error(ref_.span()));
                    }
                }
                _ => {
                    return Err(bad_type_error(typ.span()));
                }
            }
        }
        _ => {
            return Err(bad_type_error(ident.span()));
        }
    }

    Ok(())
}

fn func_check(func: &ItemFn) -> Result<(), TokenStream> {
    sig_check(func)?;

    
    Ok(())
}

pub fn to_jit_inner(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_args = match NestedMeta::parse_meta_list(attr.into()) {
        Ok(v) => v,
        Err(e) => {
            return TokenStream::from(Error::from(e).write_errors());
        }
    };

    let mut func = syn::parse_macro_input!(item as ItemFn);

    let args = match ToJitArgs::from_list(&attr_args) {
        Ok(v) => v,
        Err(e) => {
            return TokenStream::from(e.write_errors());
        }
    };

    // Check that the function is a Module type function (&Tensor)->Tensor
    match func_check(&func) {
        Ok(_) => {}
        Err(e) => {
            return e;
        }
    }

    let type_ident = args.kind.to_ident();
    let tensor_size = args.tensor_size
        .into_iter()
        .map(|x|x as i64).collect::<Vec<_>>();

    let ident = func.sig.ident.clone();
    let name = ident.to_string();
    let mod_ident = syn::Ident::new(&format!("{}_jit", ident), ident.span());
    let inner_ident = syn::Ident::new(&format!("{}_jitless", ident), ident.span());

    let outer_sig = func.sig.clone();
    let outer_vis = func.vis.clone();
    func.sig.ident = inner_ident.clone();
    func.vis = syn::Visibility::Public(Pub::default());

    let param_ident = match &outer_sig.inputs[0] {
        syn::FnArg::Typed(typed) => {
            let pat = &typed.pat;
            match pat.as_ref() {
                syn::Pat::Ident(ident) => ident.ident.clone(),
                _ => unreachable!(),
            }
        },
        _ => unreachable!()
    };

    quote::quote!(
        mod #mod_ident {
            use lazy_static::lazy_static;
            use std::sync::RwLock;
            use std::collections::HashMap;

            use tch::{Tensor, Device, Kind};
            use tch::nn::Module;
            use tch::jit::CModule;
            
            lazy_static!{
                pub static ref MODULES: RwLock<HashMap<Device, CModule>> = RwLock::new(HashMap::new());
            }

            #func
        }

        #outer_vis #outer_sig{
            let device = #param_ident.device();
            // If the module is not already created, create it
            if !#mod_ident::MODULES.read().unwrap().contains_key(&device){
                let mut modules = #mod_ident::MODULES.write().unwrap();
                modules.entry(device).or_insert_with(|| {
                    let input = Tensor::randn(&[#(#tensor_size),*], (Kind::#type_ident, device));
                    let mut module = CModule::create_by_tracing(
                        #name,
                        "forward",
                        &[input],
                        &mut|x| x
                            .iter()
                            .map(#mod_ident::#inner_ident)
                            .collect::<Vec<_>>()
                        ).unwrap();
                    module.to(device, Kind::Float, true);
                    module
                });
                
            }
            let modules = #mod_ident::MODULES.read().unwrap();
            let module = modules.get(&device).unwrap();
            module.forward(#param_ident)
        }
    ).into()
}
