import React, {useCallback, useEffect, useRef, useState} from 'react'
import {
  ConfigProvider,
  Input,
  Image,
  Row,
  Col,
  Button,
  Typography,
  Modal,
  Form,
  Slider,
  InputNumber,
  Select,
  Card,
  notification,
  Space
} from 'antd'
import {Dices, Settings2} from "lucide-react";
import get from 'lodash/get'


const API_URL = process.env.NODE_ENV === 'development'
    ? 'http://127.0.0.1:5000/api/predict'
    : window.location.origin + '/api/predict'

function App() {
  const [amount, setAmount] = useState(5)
  const [top, setTop] = useState(10)
  const [uniform, setUniform] = useState(false)
  const [maxLen, setMaxLen] = useState(24)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [names, setNames] = useState<Array<string>>([])
  const ref = useRef(null)

  const handleAmountChange = useCallback((value: number | null) => {
    if (value) {
      setAmount(value)
    }
  }, [])

  const handleTopChange = useCallback((value: number | null) => {
    if (value) {
      setTop(value)
    }
  }, [])

  const handleMaxLenChange = useCallback((value: number | null) => {
    if (value) {
      setMaxLen(value)
    }
  }, [])

  const handleOpenSettings = useCallback(() => {
    setSettingsOpen(true)
  }, [])

  const handleCloseSettings = useCallback(() => {
    setSettingsOpen(false)
  }, [])

  useEffect(() => {
    if (loading) {

    }
  }, [loading])

  const handleFetch = async () => {
    setLoading(true)

    const url = new URL(API_URL)
    url.searchParams.set('amount', String(amount))
    url.searchParams.set('maxLen', String(maxLen))
    url.searchParams.set('top', String(top))
    if (uniform) {
      url.searchParams.set('uniform', String(uniform))
    }
    const prefix = get(ref, ['current', 'input', 'defaultValue'])
    if (prefix) {
      url.searchParams.set('prefix', prefix)
    }

    try {
      const response = await fetch(url, { method: 'POST' })
      if (!response.ok) {
        const errorMessage = await response.text()
        throw new Error(errorMessage)
      }

      const result = await response.json()
      setNames(result)

    } catch (err) {
      notification.error({ message: get(err, 'message') || 'Failed to fetch' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <ConfigProvider theme={{
      token: {
        colorPrimary: '#222',
        controlHeightLG: 40,
        fontSize: 16,
        borderRadius: 0,
        controlOutlineWidth: 0,
        colorBorder: '#222',
        colorBorderSecondary: '#222',
        lineWidth: 2,
      }
    }}>
      <Settings2
        size={40}
        style={{ position: "absolute", top: 20, right: 16, cursor: 'pointer', zIndex: 1000 }}
        onClick={handleOpenSettings}
      />
      <Row gutter={[40, 40]} style={{ paddingBottom: 60 }}>
        <Col span={24}>
          <Image
              src={'/logo.png'}
              preview={false}
              alt={'logo'}
               style={{ width: '100%', height: 120, objectFit: 'contain', objectPosition: 'center' }}
               wrapperStyle={{ width: '100%', marginTop: 200 }}/>
        </Col>
        <Col span={24}>
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Input
                  placeholder={'Name starts with...'}
                  size={'large'}
                  style={{ width: '100%' }}
                  ref={ref}
              />
            </Col>
            <Col span={24}>
              <Button
                  type={'primary'}
                  block
                  size={'large'}
                  icon={<Dices color={'white'}/>}
                  loading={loading}
                  onClick={handleFetch}
              >
                <Typography.Title
                    style={{ color: 'white', margin: 0, display: 'inline' }}
                    level={4}>
                  Generate
                </Typography.Title>
              </Button>
            </Col>
          </Row>
        </Col>
        <Col span={24}>
          {Boolean(names.length) && (
              <Card>
                <Space direction={"vertical"} size={16}>
                  {names.map((name, idx) => (
                      <Typography.Text style={{ fontSize: 20 }} key={idx}>{name}</Typography.Text>
                  ))}
                </Space>
              </Card>
          )}
        </Col>
      </Row>
      <Modal
          open={settingsOpen}
          title={'Settings'}
          cancelText={null}
          onCancel={handleCloseSettings}
          onOk={handleCloseSettings}
          okButtonProps={{ size: 'large' }}
          cancelButtonProps={{ style: { display: 'none' } }}
      >
        <Form
          layout={'vertical'}
          size={'large'}
          labelWrap
        >
          <Form.Item label={'Results per request:'}>
            <Row gutter={[16, 16]}>
              <Col span={16}>
                <Slider min={1} max={10} value={amount} onChange={handleAmountChange}/>
              </Col>
              <Col span={8} style={{ display: 'flex', flexDirection: 'row-reverse'}}>
                <InputNumber min={1} max={10} value={amount} onChange={handleAmountChange}/>
              </Col>
            </Row>
          </Form.Item>
          <Form.Item label={`On each step select from top ${top} variants:`}>
            <Row gutter={[16, 16]}>
              <Col span={16}>
                <Slider min={1} max={27} value={top} onChange={handleTopChange}/>
              </Col>
              <Col span={8} style={{ display: 'flex', flexDirection: 'row-reverse'}}>
                <InputNumber min={1} max={27} value={top} onChange={handleTopChange}/>
              </Col>
            </Row>
          </Form.Item>
          <Form.Item label={'From this top variants select new letter:'}>
            <Row gutter={[16, 16]}>
              <Select
                style={{ marginLeft: 8, width: 'calc(100% - 16px)' }}
                defaultValue={false}
                onChange={setUniform}
                value={uniform}
                options={[
                  { value: true, label: 'With equal probability' },
                  { value: false, label: 'According to their probabilities' },
                ]}
              />
            </Row>
          </Form.Item>
          <Form.Item label={`Max name length:`}>
            <Row gutter={[16, 16]}>
              <Col span={16}>
                <Slider min={5} max={50} value={maxLen} onChange={handleMaxLenChange}/>
              </Col>
              <Col span={8} style={{ display: 'flex', flexDirection: 'row-reverse'}}>
                <InputNumber min={5} max={50} value={maxLen} onChange={handleMaxLenChange}/>
              </Col>
            </Row>
          </Form.Item>
        </Form>
      </Modal>
    </ConfigProvider>
  );
}

export default App;
