const { remote } = require('webdriverio');

function delay(t, val) {
   return new Promise(function(resolve) {
       setTimeout(function() {
           resolve(val);
       }, t);
   });
}
async function getPriceDelayed(browser, time) {

  return new Promise(resolve => {
    console.log('current time1:' +  Date.now())
    setTimeout(async function(browser) {
      console.log('current time before fetching:' +  Date.now())
      let result = await browser.executeAsync(done => {
        function get(oReq, url, callback) {
          oReq.open("GET", url)
          oReq.onreadystatechange = function(e) {
              console.log(oReq.readyState)
              if (oReq.readyState === 4) {
                  console.log('price')
                  console.log(oReq.responseText)
                  callback()
              }
          }
          oReq.onerror = function() {
              console.log('error!')
              
          }
          oReq.send()
        }

        function post(oReq, url, form, callback) {
          oReq.open("POST", url)
          oReq.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
          oReq.onreadystatechange = function(e) {
            if (oReqList[1].readyState === 4) {
              console.log('queue')
              console.log(oReq.responseText)
              callback()
            }
          }
          oReq.send('identifier=101&marketplace=11&orderdjupsantal=1&country=Sverige')
        }


        let baseUrl = 'https://www.nordnet.se/'
        let urlList = ['graph/instrument/11/101/price',
                        'mux/ajax/marknaden/aktiehemsidan/orderdjup.html']
        
        var responseCount = 0


        
        oReqList = []
        for (let i = 0; i < urlList.length; i++) {
          let oReq = new XMLHttpRequest();
          oReqList.push(oReq)

          url = urlList[i]
          if (i === 0) {
              get(oReq, url, () => {console.log('get callback')})
          } else {
              form = 'identifier=101&marketplace=11&orderdjupsantal=1&country=Sverige'
              post(oReq, url, form,
                  () => {console.log('post callback')})

          }
        }
      })
      console.log('current time after fetching:' +  Date.now())
      console.log(result)
      resolve(result)
    }, time, browser)
  })
}

// TODO:
// https://www.nordnet.se/mux/ajax/marknaden/aktiehemsidan/orderdjup.html
/*
<h2 antal="1">Orderdjup

<div class="iconTank">
    <a onclick="new Ajax.Updater('orderdjup', '/mux/ajax/marknaden/aktiehemsidan/orderdjup.html', {parameters:'identifier=100&marketplace=11&orderdjupsantal=1&country=Sverige'}); return false" href="#"><img src="/now/images/knapp_refresh2.gif" alt="Uppdatera" title="Uppdatera" /></a>
</div>

</h2>






<table class="borders">
    <thead>
        <tr>
            <th><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_antal', event)">Antal</a></th>
            <th><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_pris', event)">Pris</a></th>
            <th class="buy"><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_k%f6p', event)">Köp</a></th>
            <th class="sell"><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_s%e4lj', event)">Sälj</a></th>
            <th><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_pris', event)">Pris</a></th>
            <th><a href="#" style="text-decoration:none;color:#018acf;" onclick="return openHjalpOrd( 'orderdjup_antal', event)">Antal</a></th>
        </tr>
    </thead>
    <tr>
        <td class="first">558</td>
            <td>95,80</td>
        <td colspan="2" class="orderdjupStaplar">
            <div class="stapelContainer">
                <div class="stapelUpp" style="width:58px;"><!-- IE friendly --></div>
                <div class="stapelNer" style="width:78px;"><!-- IE friendly --></div>
            </div>
        </td>
            <td>95,80</td>
        <td class="last">749</td>
    </tr>

</table>


const $ = cheerio.load(html)
for (i=0; i<2; i++) {
    console.log(i)
    const row = $(`.borders tr:nth-of-type(${i+1})`)
    //console.log(firstRow)
    const vol = $(row).find('td:nth-of-type(1)').text()
    console.log(vol)
    const prise = $(row).find('td:nth-of-type(2)').text()
    console.log(prise)
}




*/

(async () => {
/*
    var browser = await remote({
        logLevel: 'trace',
        host: '0.0.0.0',
        port: 4444,
        //path: '/', // only for firefox
        capabilities: {
            browserName: 'chrome'
        }
    });
*/
    
    var browser = await remote({
        logLevel: 'info',
        host: 'localhost',
        port: 4321,
        //path: '/', // only for firefox
        capabilities: {
            browserName: 'chrome'
        }
    });
    

    /*
    var browser = await remote({
        logLevel: 'info',
        host: 'localhost',
        port: 4445,
        path: '/', // only for firefox
        capabilities: {
            browserName: 'firefox'
        }
    });
    */
    await browser.url('https://www.nordnet.se/start.html')
    await delay(10000)
    await browser.setTimeout({ 'script': 60000 });
    while (1) {
        // get time delta to next minute.
        msNow = Date.now()
        msToNextMinute = 60000 - Date.now() % 60000
        msToNextMinute -= 20
        console.log(`current time: ${msNow}, wait: ${msToNextMinute}`)
        await getPriceDelayed(browser, msToNextMinute)

    }
 

    /*
    const link = await browser.$('=Logga in')
    //console.log(link)
    text = await link.getText()
    console.log(text)

    await link.click()
    await delay(500)

    // find out the Mobilt bankid
    const mobiltBankId = await browser.$('.button-1-0.primary-1-6.size-m-1-19.block-1-1')
    console.log(await mobiltBankId.getText())

    await mobiltBankId.click()
    await delay(500)

    const id = await browser.$('.text-input.id-number-input')
    await id.setValue('197705103815')
    await delay(500)

    const ok = await browser.$('.ok.large-button.button')
    await ok.click()

    await delay(10000)
   // TODO: login

    */


    const title = await browser.getTitle();
    console.log('Title was: ' + title);
    
    await browser.deleteSession();
})().catch((e) => console.error(e));
