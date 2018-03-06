window.onload=()=>{
    let rf_time=1000
    async function get_svg()
    {
        let res=await fetch("/img")
        let text=await res.text()
        let ele=document.querySelector("#view")
        ele.innerHTML=text
        setTimeout(get_svg,rf_time)
    }
    setTimeout(get_svg,rf_time)
}
