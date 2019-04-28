const { remote } = require('webdriverio');

function delay(t, val) {
   return new Promise(function(resolve) {
       setTimeout(function() {
           resolve(val);
       }, t);
   });
}


(async () => {

    var browser = await remote({
        logLevel: 'error',
        host: 'localhost',
        port: 4321,
        //path: '/', // only for firefox
        capabilities: {
            browserName: 'chrome'
        }
    });

    await browser.url('https://www.nordnet.se');

    await delay(5000)
    await browser.setTimeout({ 'script': 60000 });
    let result = await browser.executeAsync(function(a, done) {
        console.log(a)
        var oReq = new XMLHttpRequest();

        oReq.open("GET", "https://www.nordnet.se/graph/instrument/11/101/price")
        oReq.onreadystatechange = function(e) {
            if (oReq.readyState === 4) {
                console.log('event triggered ' +  oReq.responseText)
                console.log(typeof(oReq.responseText))
                str = (' ' + oReq.responseText).slice(1);
                console.log(str)
                //str = '{"open":95.0,"close":94.32,"low":94.52,"high":95.4,"last":95.24,"decimals":2}'
                //done('{"open":95.0,"close":94.32,"low":94.52,"high":95.4,"last":95.24,"decimals":2}')
                done(oReq.responseText)
            }
 
        };
        oReq.onerror = function() {
            console.log('error!')
            done('error')
        }
        oReq.send();
        
    }, 100)
    console.log('finished')
    console.log(result)
    await delay(100000)
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



    const title = await browser.getTitle();
    console.log('Title was: ' + title);
    */
    await browser.deleteSession();
})().catch((e) => console.error(e));
