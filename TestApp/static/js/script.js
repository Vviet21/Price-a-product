const searchWrapper=document.querySelector(".search-input");
const inputBox=searchWrapper.querySelector("input");
const suggBox=searchWrapper.querySelector(".autocom-box");


inputBox.onkeyup = (e)=>{
    let userData = e.target.value;
    let emptyArray = [];
    if(userData){
        emptyArray = suggestion.filter((data) =>{
            return data.toLocaleLowerCase().startsWith(userData.toLocaleLowerCase());
            
        })
        emptyArray = emptyArray.map((data)=>{
            return data = '<li>'+data+'</li>' ;
        })
        console.log(emptyArray)
        searchWrapper.classList.add("active");
        showSugg(emptyArray)
        let allList = suggBox.querySelectorAll("li");
        for(let i = 0 ; i < allList.length;i++){
            allList[i].setAttribute("onclick","select(this)");
        }
    }else{
        searchWrapper.classList.remove("active");

    }
}
function select(element){
    let selectUserdata = element.textContent;
    inputBox.value = selectUserdata;
    searchWrapper.classList.remove("active");

    console.log(inputBox.value)

}
function showSugg(list){
    let listData;
    if (!list.length){
        uservalue = inputBox.value;
        listData = '<li>'+uservalue+'</li>';
    }
    else{
        listData =list.join('');
    }
    suggBox.innerHTML =listData;
}